"""量产音频目录查询：标签词典 + 音频原料。"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from bson import ObjectId
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.cache.somni_audio_catalog_cache import (
    SomniAudioCatalogCache,
    catalog_cache_key,
    clear_somni_audio_catalog_cache,
)
from app.core.audio_tags import (
    canonical_disliked_tags,
    canonical_tags,
    material_is_excluded,
    material_tag_codes,
    reference_tag,
    resolve_audio_preferences,
    validated_audio_tags,
    validated_content_references,
)
from app.core.bson_util import bson_to_jsonable
from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError, EncoderNotReadyError
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.es.somni_docs import resolve_query_language, resolve_search_language
from app.schemas.audio import SearchAudioRequest
from app.schemas.search_material import project_search_material
from app.server.somni.audio.hot import HotTracker, normalize_tag_code
from app.server.somni.audio.relevance_filter import (
    apply_lane_relevance_filters,
    detect_color_noise_intents,
    diversify_multi_color_noise,
    filter_conflicting_color_noise,
)
from app.server.somni.audio.rrf import rrf_merge
from app.server.somni.audio.segment import (
    segment_dislike_phrases,
    segment_dislike_text,
    segment_dislike_text_en,
)

_TAG_ENABLED = "启用"
_CONTENT_FORM = "content_form"


_material_snapshot: ContextVar[dict[str, dict[str, Any] | None] | None] = ContextVar(
    "prescription_material_snapshot", default=None
)


@contextmanager
def prescription_material_scope() -> Iterator[None]:
    """One authoritative Mongo read per distinct material within a batch RPC."""
    token = _material_snapshot.set({})
    try:
        yield
    finally:
        _material_snapshot.reset(token)


class InvalidAudioQueryError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=HttpStatus.BAD_REQUEST)


class AudioCatalogService:
    def __init__(
        self,
        client: AsyncIOMotorClient | None,
        settings: Settings,
        *,
        es_search: EsSearch | None = None,
        encoder: Encoder | None = None,
        hot: HotTracker | None = None,
        retrieval: Any | None = None,
        redis: Any | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._es_search = es_search
        self._encoder = encoder
        self._hot = hot
        self._content_form_tag_bank_cache: list[dict[str, Any]] | None = None
        self._retrieval = retrieval
        self._catalog_cache = (
            SomniAudioCatalogCache(redis, ttl_sec=settings.somni_audio_catalog_cache_ttl_sec)
            if redis is not None
            else None
        )
        self._catalog_load_lock = asyncio.Lock()
        self._hot_tasks: set[asyncio.Task[None]] = set()
        self._hot_sem = asyncio.Semaphore(32)

    async def get_audio_tag(self) -> dict[str, Any]:
        collection = self._tags()
        query = _root_tag_query()
        total = await collection.count_documents(query)
        self._reject_over_limit(total)
        cursor = collection.find(
            query,
            {
                "_id": 1,
                "id": 1,
                "type": 1,
                "code": 1,
                "name": 1,
                "name_en": 1,
                "parent_tag_id": 1,
                "status": 1,
            },
        )
        docs = [bson_to_jsonable(doc) async for doc in cursor]
        return {"tags": [_map_tag_dict(doc) for doc in docs]}

    async def get_audio(
        self,
        *,
        page: int | None,
        page_size: int | None,
        fetch_all: bool,
        query_text: str,
        tag_code: str,
        language: str = "zh",
        content_form_codes: list[str] | None = None,
        preference_text: str = "",
        dislike_text: str = "",
    ) -> dict[str, Any]:
        text = query_text.strip()
        try:
            code = normalize_tag_code(tag_code)
            codes = _normalize_content_form_codes(content_form_codes)
        except ValueError as exc:
            raise InvalidAudioQueryError(
                "unknown_audio_tag: use reviewed English dictionary codes"
            ) from exc
        pref = preference_text.strip()
        dislike = dislike_text.strip()
        language = resolve_search_language(language)
        # 有搜索词：ES 双路（BM25 混合文本 + knn 混合向量），不套问卷偏好。
        if text and self._search_path_enabled():
            return await self._get_audio_hybrid_search(
                text=text,
                tag_code=code,
                language=language,
                fetch_all=fetch_all,
                page=page,
                page_size=page_size,
            )
        # 列表 / 标签筛选：读全量启用音频，再按请求 language 软过滤（空串或同语言）。
        docs = await self._load_audios(from_es=True, language="all")
        catalog_size = len(docs)
        docs = _filter_docs_by_request_language(docs, language)
        hot_tag_codes: set[str] = set()
        if code:
            docs = [doc for doc in docs if _has_content_form_code(doc, code)]
            hot_tag_codes.add(code)
        preferred_ids: set[str] = set()
        if codes or pref:
            preferred_ids = await self._preferred_audio_ids(
                docs,
                codes=codes,
                preference_text=pref,
            )
            docs = _order_preferred_then_rest(docs, preferred_ids)
        disliked_ids = await self._disliked_audio_ids(
            docs,
            dislike,
            catalog_size=catalog_size,
        )
        docs = _move_disliked_to_end(docs, disliked_ids)
        payload = _paginate_docs(docs, page, page_size, fetch_all, self._settings)
        payload["list"] = [
            _to_audio_list_item(item, language=language) for item in payload["list"]
        ]
        self._schedule_hot(
            query_text,
            language,
            int(payload.get("total") or 0),
            tag_codes=hot_tag_codes,
        )
        return payload

    def _search_path_enabled(self) -> bool:
        return (
            self._settings.get_audio_vector_enabled
            or self._settings.get_audio_bm25_enabled
        )

    async def _get_audio_hybrid_search(
        self,
        *,
        text: str,
        tag_code: str,
        language: str,
        fetch_all: bool,
        page: int | None,
        page_size: int | None,
    ) -> dict[str, Any]:
        es_search = self._require_es_search()
        # 检索字段/模型按输入文字语种；请求 language 只做可见性与展示
        query_language = resolve_query_language(text, language)
        query_vector, degraded = await self._prepare_search_inputs(
            text, language=query_language
        )
        bm25_docs, vector_docs, lane_degraded = await self._search_candidates(
            es_search=es_search,
            text=text,
            query_vector=query_vector,
            tag_code=tag_code,
            language=query_language,
            fetch_all=fetch_all,
        )
        for lane in lane_degraded:
            if lane not in degraded:
                degraded.append(lane)
        self._raise_if_all_lanes_failed(degraded)
        bm25_docs = _filter_docs_by_request_language(bm25_docs, language)
        vector_docs = _filter_docs_by_request_language(vector_docs, language)
        color_intents = detect_color_noise_intents(text)
        multi_color = len(color_intents) >= 2
        cliff_ratio = self._settings.resolve_get_audio_score_cliff_ratio(query_language)
        # 多色：按色分别断崖，避免一色顶分把另一色整段砍掉
        lane_intents = color_intents if multi_color else None
        bm25_docs = apply_lane_relevance_filters(
            bm25_docs,
            score_field="_bm25_score",
            min_score=self._settings.get_audio_min_bm25_score,
            cliff_ratio=cliff_ratio,
            color_intents=lane_intents,
        )
        vector_docs = apply_lane_relevance_filters(
            vector_docs,
            score_field="_knn_score",
            min_score=self._settings.get_audio_min_vector_score,
            cliff_ratio=cliff_ratio,
            color_intents=lane_intents,
        )
        merged_docs = _merge_hybrid_search_documents(
            bm25_docs,
            vector_docs,
            rrf_k=self._settings.get_audio_rrf_k,
        )
        if multi_color:
            merged_docs = diversify_multi_color_noise(
                merged_docs,
                query_text=text,
                per_color=2,
            )
        elif self._settings.get_audio_color_noise_mutex_enabled:
            merged_docs = filter_conflicting_color_noise(
                merged_docs,
                query_text=text,
            )
        mapped = [_map_material(doc) for doc in merged_docs]
        if not fetch_all:
            mapped = mapped[: self._settings.get_audio_search_top_k]
        payload = _paginate_docs(mapped, page, page_size, fetch_all, self._settings)
        payload["list"] = [
            _to_audio_list_item(item, language=language) for item in payload["list"]
        ]
        self._schedule_hot(
            text,
            language,
            int(payload.get("total") or 0),
            tag_codes={tag_code} if tag_code else set(),
        )
        if degraded:
            payload["degraded"] = list(degraded)
        return payload

    async def _prepare_search_inputs(
        self, text: str, *, language: str
    ) -> tuple[list[float] | None, list[str]]:
        degraded: list[str] = []
        query_vector, encode_failed = await self._encode_query_if_needed(
            text, language=language
        )
        if encode_failed:
            degraded.append("vector")
        return query_vector, degraded

    async def _encode_query_if_needed(
        self, text: str, *, language: str
    ) -> tuple[list[float] | None, bool]:
        if not self._settings.get_audio_vector_enabled:
            return None, False
        try:
            vector = await self._require_encoder().encode_one(text, language=language)
            return vector, False
        except Exception as exc:  # noqa: BLE001 — 编码失败不阻断 BM25
            logger.warning("GetAudio 向量编码失败，降级跳过 knn：{}", exc)
            return None, True

    def _require_encoder(self) -> Encoder:
        if self._encoder is None or not self._encoder.is_loaded:
            raise EncoderNotReadyError()
        return self._encoder

    def _require_es_search(self) -> EsSearch:
        if self._es_search is None:
            raise AppError(
                message="Elasticsearch 未就绪，无法按搜索词查询音频",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        return self._es_search

    async def _search_candidates(
        self,
        *,
        es_search: EsSearch,
        text: str,
        query_vector: list[float] | None,
        tag_code: str,
        language: str,
        fetch_all: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        jobs: list[tuple[str, Any]] = []
        if self._settings.get_audio_bm25_enabled:
            jobs.append(
                (
                    "bm25",
                    es_search.search_get_audio_bm25(
                        text,
                        language=language,
                        tag_code=tag_code,
                        size=self._bm25_recall_size(fetch_all),
                    ),
                )
            )
        if self._settings.get_audio_vector_enabled and query_vector is not None:
            jobs.append(
                (
                    "vector",
                    self._vector_search(
                        es_search,
                        query_vector,
                        tag_code=tag_code,
                        language=language,
                        fetch_all=fetch_all,
                    ),
                )
            )
        return await self._gather_search_jobs(jobs)

    def _bm25_recall_size(self, fetch_all: bool) -> int:
        if fetch_all:
            return self._settings.fetch_all_hard_limit
        return max(
            self._settings.get_audio_search_top_k,
            self._settings.get_audio_knn_k,
            self._settings.get_audio_bm25_size,
        )

    def _vector_search(
        self,
        es_search: EsSearch,
        query_vector: list[float],
        *,
        tag_code: str,
        language: str,
        fetch_all: bool,
    ):
        if fetch_all:
            return es_search.search_get_audio_vector_all(
                query_vector,
                language=language,
                tag_code=tag_code,
                size=self._settings.fetch_all_hard_limit,
            )
        return es_search.search_get_audio_knn(
            query_vector,
            language=language,
            tag_code=tag_code,
            k=self._settings.get_audio_knn_k,
            num_candidates=self._settings.get_audio_knn_num_candidates,
        )

    async def _gather_search_jobs(
        self,
        jobs: list[tuple[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        if not jobs:
            return [], [], []
        results = await asyncio.gather(
            *(coro for _, coro in jobs),
            return_exceptions=True,
        )
        bm25_docs: list[dict[str, Any]] = []
        vector_docs: list[dict[str, Any]] = []
        degraded: list[str] = []
        for (kind, _), result in zip(jobs, results):
            if isinstance(result, BaseException):
                logger.warning("GetAudio {} 检索失败，单路降级：{}", kind, result)
                degraded.append(kind)
                continue
            if kind == "bm25":
                bm25_docs = result
            else:
                vector_docs = result
        return bm25_docs, vector_docs, degraded

    def _raise_if_all_lanes_failed(self, degraded: list[str]) -> None:
        want_bm25 = self._settings.get_audio_bm25_enabled
        want_vector = self._settings.get_audio_vector_enabled
        bm25_ok = want_bm25 and "bm25" not in degraded
        vector_ok = want_vector and "vector" not in degraded
        if bm25_ok or vector_ok:
            return
        if want_bm25 and want_vector:
            message = "音频检索暂不可用：BM25 与向量均失败"
        elif want_bm25:
            message = "音频检索暂不可用：BM25 失败"
        else:
            message = "音频检索暂不可用：向量检索失败"
        raise AppError(
            message=message,
            status_code=HttpStatus.SERVICE_UNAVAILABLE,
        )

    async def _preferred_audio_ids(
        self,
        docs: list[dict[str, Any]],
        *,
        codes: set[str],
        preference_text: str,
    ) -> set[str]:
        """勾选按英文标签 code 精确命中；自由输入保留自然语言向量召回。"""
        preferred = {
            str(doc.get("id") or "")
            for doc in docs
            if any(_has_content_form_code(doc, name) for name in codes)
        }
        preferred.discard("")
        if not preference_text:
            return preferred
        tag_ids = await self._free_text_tag_ids(preference_text)
        preferred.update(_audio_ids_for_tag_ids(docs, tag_ids))
        return preferred

    async def _disliked_audio_ids(
        self,
        docs: list[dict[str, Any]],
        dislike_text: str,
        *,
        catalog_size: int,
    ) -> set[str]:
        """讨厌的自由输入按短语切词，命中的音频排到最后。

        中文词各自并发：标签名向量（只取与最高分接近的标签）、音频描述向量，
        以及描述命中音频身上、名字也和这个词相近的二级标签。
        另有音频名和标签名的字面匹配；英文词只对标签英文名向量。
        """
        if not dislike_text.strip() or not docs:
            return set()
        try:
            tags = await self._content_form_tag_bank()
        except Exception as exc:  # noqa: BLE001 — 厌恶排序失败不阻断目录
            logger.warning("厌恶输入读取标签向量失败：{}", exc)
            tags = []
        vocabulary = [
            part
            for tag in tags
            for part in re.split(r"[/／、()（）\s-]+", str(tag.get("display_name") or ""))
        ]
        han_words = _han_words(segment_dislike_phrases(dislike_text, vocabulary))
        en_words = segment_dislike_text_en(dislike_text)
        threshold = self._settings.get_audio_root_tag_sim_threshold
        margin = self._settings.get_audio_dislike_tag_sim_margin
        found = _audio_ids_by_literal(docs, han_words)
        if en_words and tags:
            en_tag_ids = await self._tag_ids_for_words(
                tags,
                en_words,
                self._encoder,
                "vector_en",
                threshold,
                margin=margin,
                language="en",
            )
            found |= _audio_ids_for_tag_ids(docs, en_tag_ids)
        if not han_words:
            return found
        if self._encoder is None or not self._encoder.is_loaded:
            logger.warning("厌恶输入未做中文向量：编码器未就绪")
            return found
        # ES knn 的 num_candidates 上限 10000，检索侧按 size 的 3 倍取候选
        size = max(1, min(catalog_size, 3000))
        per_word = await asyncio.gather(
            *(
                self._disliked_ids_for_word(docs, tags, word, size=size)
                for word in han_words
            )
        )
        for ids in per_word:
            found |= ids
        return found

    async def _disliked_ids_for_word(
        self,
        docs: list[dict[str, Any]],
        tags: list[dict[str, Any]],
        word: str,
        *,
        size: int,
    ) -> set[str]:
        # 中文 dislike 词固定走中文模型/字段，与请求 language 无关
        try:
            query = await self._encoder.encode_one(word, language="zh")
        except Exception as exc:  # noqa: BLE001
            logger.warning("厌恶输入词向量失败：{}", exc)
            return set()
        tag_scores = _tag_scores_for_language(tags, query, "zh")
        found = _audio_ids_for_tag_ids(
            docs,
            _tag_ids_near_top(
                tag_scores,
                self._settings.get_audio_root_tag_sim_threshold,
                self._settings.get_audio_dislike_tag_sim_margin,
            ),
        )
        described = await self._audio_ids_by_description(
            docs, query, language="zh", size=size
        )
        found |= described
        spread = _spread_tag_ids(
            docs,
            described,
            tags,
            tag_scores,
            self._settings.get_audio_dislike_tag_spread_sim,
        )
        found |= _audio_ids_for_tag_ids(docs, spread)
        return found

    async def _audio_ids_by_description(
        self,
        docs: list[dict[str, Any]],
        query: list[float],
        *,
        language: str,
        size: int,
    ) -> set[str]:
        """词向量对音频描述向量，取不低于下限、且与最高分足够接近的音频。"""
        if self._es_search is None:
            return set()
        try:
            hits = await self._es_search.search_by_description_vector(
                query,
                sleep_stage_tags=[],
                size=size,
                language=language,
                include_source=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("厌恶输入描述向量检索失败：{}", exc)
            return set()
        # ES cosine 相似度的 _score 为 (1 + cos) / 2
        scored = [
            (2 * float(hit.get("_description_score") or 0.0) - 1, str(hit.get("_id") or ""))
            for hit in hits
        ]
        if not scored:
            return set()
        floor = self._settings.get_audio_dislike_desc_sim_floor
        margin = self._settings.get_audio_dislike_desc_sim_margin
        cutoff = max(floor, max(score for score, _ in scored) - margin)
        doc_ids = {str(doc.get("id") or "") for doc in docs}
        return {audio_id for score, audio_id in scored if score >= cutoff and audio_id in doc_ids}

    async def _free_text_tag_ids(self, text: str) -> set[str]:
        """每个词单独编码。中文对标签中文名向量，英文对标签英文名向量。"""
        han_words = _han_words(segment_dislike_text(text))
        en_words = segment_dislike_text_en(text)
        if not han_words and not en_words:
            return set()
        try:
            tags = await self._content_form_tag_bank()
        except Exception as exc:  # noqa: BLE001 — 偏好排序失败不阻断目录
            logger.warning("自由输入读取标签向量失败：{}", exc)
            return set()
        if not tags:
            return set()
        threshold = self._settings.get_audio_root_tag_sim_threshold
        matched: set[str] = set()
        if han_words:
            matched.update(
                await self._tag_ids_for_words(
                    tags,
                    han_words,
                    self._encoder,
                    "vector",
                    threshold,
                    language="zh",
                )
            )
        if en_words:
            matched.update(
                await self._tag_ids_for_words(
                    tags,
                    en_words,
                    self._encoder,
                    "vector_en",
                    threshold,
                    language="en",
                )
            )
        return matched

    async def _tag_ids_for_words(
        self,
        tags: list[dict[str, Any]],
        words: list[str],
        encoder: Encoder | None,
        vector_field: str,
        threshold: float,
        *,
        margin: float | None = None,
        language: str = "zh",
    ) -> set[str]:
        if encoder is None or not encoder.is_loaded:
            logger.warning(
                "自由输入未做{}向量：编码器未就绪", "英文" if language == "en" else "中文"
            )
            return set()
        matched: set[str] = set()
        for word in words:
            try:
                query = await encoder.encode_one(word, language=language)
            except Exception as exc:  # noqa: BLE001
                logger.warning("自由输入词向量失败：{}", exc)
                continue
            scores = _tag_scores(tags, query, vector_field)
            matched.update(_tag_ids_near_top(scores, threshold, margin))
        return matched

    async def _content_form_tag_bank(self) -> list[dict[str, Any]]:
        if self._content_form_tag_bank_cache is not None:
            return self._content_form_tag_bank_cache
        if self._es_search is None:
            return []
        raw = await self._es_search.list_content_tag_vectors()
        bank = [dict(tag) for tag in raw if isinstance(tag, dict) and _is_content_form_dict(tag)]
        self._content_form_tag_bank_cache = bank
        return bank

    async def search_audio(self, request: SearchAudioRequest) -> list[dict[str, Any]]:
        """与 POST /api/audio/search 相同的检索流水线，数据源是量产 ES。"""
        if self._retrieval is None:
            raise AppError(message="检索服务未就绪", status_code=HttpStatus.SERVICE_UNAVAILABLE)
        return await self._retrieval.search(request)

    async def search_prescription_audio(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Use the configured ranked retrieval service without recording hot queries."""
        if self._retrieval is None:
            raise AppError(message="检索服务未就绪", status_code=HttpStatus.SERVICE_UNAVAILABLE)
        try:
            machine_tags = {
                field: (
                    canonical_disliked_tags(task.get(field) or [])
                    if field == "disliked_tags"
                    else _prescription_content_tags(task.get(field) or [])
                    if field == "content_tags"
                    else validated_audio_tags(
                        task.get(field) or [],
                        allowed_types=(
                            frozenset({"sleep_stage"})
                            if field == "sleep_stage_tags"
                            else frozenset({"content_form"})
                        ),
                    )
                )
                for field in (
                    "content_tag_codes",
                    "content_tags",
                    "sleep_stage_tags",
                    "disliked_tags",
                )
            }
        except ValueError as exc:
            raise InvalidAudioQueryError(
                "unknown_audio_tag: use reviewed English dictionary codes"
            ) from exc
        codes = machine_tags["content_tag_codes"]
        resolved: dict[str, str] = {}
        if codes:
            cursor = self._tags().find(
                {
                    "code": {"$in": codes},
                    "type": "content_form",
                    "status": _TAG_ENABLED,
                },
                {"code": 1, "type": 1, "name": 1},
            )
            matches: dict[str, set[tuple[str, str]]] = {code: set() for code in codes}
            async for row in cursor:
                code = str(row.get("code") or "")
                tag_type = str(row.get("type") or "")
                label = str(row.get("name") or "")
                if code and tag_type and label:
                    matches.setdefault(code, set()).add((tag_type, label))
            if any(len(matches.get(code, set())) != 1 for code in codes):
                # An unresolved code must not be replaced with an approximate tag.
                return []
            resolved = {code: next(iter(matches[code]))[1] for code in codes}
        request = SearchAudioRequest(
            query_text=str(task.get("query_text") or ""),
            sleep_stage_tags=machine_tags["sleep_stage_tags"],
            content_tags=canonical_tags([*resolved.keys(), *machine_tags["content_tags"]]),
            content_tag_codes=codes,
            disliked_tags=machine_tags["disliked_tags"],
            top_k=max(1, min(int(task.get("top_k") or 5), 20)),
        )
        requested_top_k = request.top_k
        request.top_k = min(60, requested_top_k * 3)
        materials = await self._retrieval.search(request)
        materials = await self._verify_prescription_materials(materials)
        materials = [
            material
            for material in materials
            if not material_is_excluded(material, request.disliked_tags)
        ]
        stages = set(request.sleep_stage_tags)
        if not stages:
            return []
        return [
            item
            for item in materials
            if stages & material_tag_codes({"sleep_stage_tags": item.get("sleep_stage_tags", [])})
        ][:requested_top_k]

    async def _verify_prescription_materials(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Mongo presence/status and safety metadata are authoritative over ES and caches."""
        identifiers = {str(item.get("_id") or item.get("id") or "") for item in candidates}
        identifiers.discard("")
        if not identifiers:
            return []
        current = _material_snapshot.get()
        if current is None:
            current = {}
        missing = identifiers - current.keys()
        if missing:
            ids: list[Any] = list(missing)
            ids.extend(ObjectId(value) for value in missing if ObjectId.is_valid(value))
            fields = (
                "_id",
                "status",
                "audio_name",
                "description",
                "audio_url",
                "cover_url",
                "sleep_stage_tags",
                "content_form_tags",
                "audio_engineering_tags",
                "evidence_level_tags",
                "duration_seconds",
                "acoustic_metrics",
                "language",
                "languages",
                "short_description",
                "source",
                "created_at",
                "updated_at",
            )
            cursor = self._materials().find(
                {"_id": {"$in": ids}, "status": True}, {key: 1 for key in fields}
            )
            fetched = {
                str(row["_id"]): project_search_material(bson_to_jsonable(row))
                async for row in cursor
            }
            current.update({identifier: fetched.get(identifier) for identifier in missing})
        verified = []
        keys = (
            "sleep_stage_tags",
            "content_form_tags",
            "audio_engineering_tags",
        )
        for candidate in candidates:
            identifier = str(candidate.get("_id") or candidate.get("id") or "")
            source = current.get(identifier)
            if source is None or source.get("status") is not True or not source.get("audio_url"):
                continue
            try:
                validated_content_references(source)
            except ValueError as exc:
                # IDs/codes only: never log URLs, titles or user preferences.
                refs = source.get("content_form_tags") or []
                codes = [str(ref.get("code") or "") for ref in refs if isinstance(ref, dict)]
                logger.warning(
                    "prescription_material_rejected audio_id={} reason={} content_codes={}",
                    identifier,
                    str(exc).split(":", 1)[0],
                    codes,
                )
                continue
            projected = project_search_material({**candidate, "_id": identifier})
            if any(projected[key] != source[key] for key in keys):
                # Changed eligibility needs fresh indexing and ranking.
                continue
            verified.append(
                {
                    **source,
                    **{key: candidate[key] for key in ("_score", "score") if key in candidate},
                }
            )
        return verified

    async def get_hot(
        self,
        *,
        language: str = "zh",
        kind: str = "query",
    ) -> dict[str, Any]:
        if self._hot is None:
            raise AppError(
                message="量产 Redis 未配置，无法获取热点",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        return {"items": await self._hot.list_hot(language=language, kind=kind)}

    def _schedule_hot(
        self,
        query_text: str,
        language: str,
        hit_count: int,
        *,
        tag_codes: set[str] | None = None,
    ) -> None:
        if self._hot is None:
            return
        if not query_text.strip() and not tag_codes:
            return
        task = asyncio.create_task(
            self._record_hot_safely(
                query_text,
                language,
                hit_count,
                tag_codes=tag_codes or set(),
            )
        )
        self._hot_tasks.add(task)
        task.add_done_callback(self._hot_tasks.discard)

    async def drain_hot_tasks(self, *, timeout_sec: float = 5.0) -> None:
        """关闭前排空热点记账任务，避免访问已关闭的 Redis/ES。"""
        pending = [task for task in self._hot_tasks if not task.done()]
        if not pending:
            return
        done, still = await asyncio.wait(pending, timeout=max(0.1, timeout_sec))
        for task in still:
            task.cancel()
        if still:
            await asyncio.gather(*still, return_exceptions=True)
            logger.warning("量产热点后台任务关闭超时，已取消 {} 个", len(still))
        _ = done

    async def _record_hot_safely(
        self,
        query_text: str,
        language: str,
        hit_count: int,
        *,
        tag_codes: set[str],
    ) -> None:
        async with self._hot_sem:
            try:
                await self._hot.record_search(
                    query_text,
                    language=language,
                    hit_count=hit_count,
                    tag_codes=tag_codes,
                )
            except Exception as exc:
                logger.warning("量产热点后台记账失败：{}", exc)

    async def warm_audio_catalog(self) -> int:
        """从 ES 覆盖写入 GetAudio 正在使用的目录缓存，并删掉不再读取的目录键。"""
        if self._catalog_cache is None:
            logger.info("未配置量产 Redis，跳过 GetAudio 目录缓存")
            return 0
        if self._es_search is not None and not await self._es_search.audio_index_exists():
            raise RuntimeError(
                f"量产音频索引不存在：{self._es_search.audio_index}，"
                "请先运行 scripts/sync_es_from_somni.py 同步数据"
            )
        language = "all"
        raw = await self._fetch_audios_es(language)
        docs = [_map_material(doc) for doc in raw]
        written = await self._catalog_cache.set(from_es=True, language=language, docs=docs)
        if not written:
            raise RuntimeError("写入量产 GetAudio 目录缓存失败")
        removed = await clear_somni_audio_catalog_cache(
            self._catalog_cache._redis,
            keep=catalog_cache_key(from_es=True, language=language),
        )
        logger.info("已重建量产 GetAudio 目录缓存，条数={}，删除过期键={}", len(docs), removed)
        return len(docs)

    async def _load_audios(self, *, from_es: bool, language: str) -> list[dict[str, Any]]:
        if self._catalog_cache is not None:
            cached = await self._catalog_cache.get(from_es=from_es, language=language)
            if cached is not None:
                return cached
        async with self._catalog_load_lock:
            if self._catalog_cache is not None:
                cached = await self._catalog_cache.get(from_es=from_es, language=language)
                if cached is not None:
                    return cached
            raw = (
                await self._fetch_audios_es(language)
                if from_es
                else await self._fetch_audios_mongo(language)
            )
            docs = [_map_material(doc) for doc in raw]
            if self._catalog_cache is not None:
                await self._catalog_cache.set(from_es=from_es, language=language, docs=docs)
            return docs

    async def _fetch_audios_mongo(self, language: str) -> list[dict[str, Any]]:
        del language  # 目录不再按请求 language 过滤
        collection = self._materials()
        query = {"status": True}
        limit = self._settings.fetch_all_hard_limit + 1
        cursor = collection.find(query, _AUDIO_LIST_PROJECTION, limit=limit)
        docs = [bson_to_jsonable(doc) async for doc in cursor]
        self._reject_over_limit(len(docs))
        return docs

    async def _fetch_audios_es(self, language: str) -> list[dict[str, Any]]:
        if self._es_search is None:
            raise AppError(
                message="Elasticsearch 未就绪，无法查询音频",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        docs = await self._es_search.list_audio_catalog_docs(
            size=self._settings.fetch_all_hard_limit + 1,
            language=language,
        )
        self._reject_over_limit(len(docs))
        return docs

    def _tags(self) -> AsyncIOMotorCollection:
        return self._db()[self._settings.somni_mongo_tag_dictionary_collection]

    def _materials(self) -> AsyncIOMotorCollection:
        return self._db()[self._settings.somni_mongo_materials_collection]

    def _db(self):
        if self._client is None:
            raise AppError(
                message="量产 Mongo 未配置（SOMNI_MONGO_URI），无法查询音频",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        return self._client[self._settings.somni_mongo_db]

    def _reject_over_limit(self, total: int) -> None:
        limit = self._settings.fetch_all_hard_limit
        if total > limit:
            raise InvalidAudioQueryError(f"全量条数超过上限 {limit}")


def _merge_hybrid_search_documents(
    bm25_docs: list[dict[str, Any]],
    vector_docs: list[dict[str, Any]],
    *,
    rrf_k: int,
) -> list[dict[str, Any]]:
    ranked_ids = [_hybrid_document_ids(bm25_docs), _hybrid_document_ids(vector_docs)]
    merged = rrf_merge(ranked_ids, k=rrf_k)
    docs_by_id: dict[str, dict[str, Any]] = {}
    for doc in bm25_docs:
        doc_id = _hybrid_document_id(doc)
        if not doc_id:
            continue
        docs_by_id[doc_id] = dict(doc)
    for doc in vector_docs:
        doc_id = _hybrid_document_id(doc)
        if not doc_id:
            continue
        existing = docs_by_id.get(doc_id)
        if existing is None:
            docs_by_id[doc_id] = dict(doc)
            continue
        if "_knn_score" in doc:
            existing["_knn_score"] = doc["_knn_score"]
        if "_bm25_score" in doc and "_bm25_score" not in existing:
            existing["_bm25_score"] = doc["_bm25_score"]
    return [
        {**docs_by_id[doc_id], "_rrf_score": rrf_score}
        for doc_id, rrf_score in merged
        if doc_id in docs_by_id
    ]


def _hybrid_document_ids(docs: list[dict[str, Any]]) -> list[str]:
    return [doc_id for doc in docs if (doc_id := _hybrid_document_id(doc))]


def _hybrid_document_id(doc: dict[str, Any]) -> str:
    return str(doc.get("_id") or doc.get("id") or "")


_AUDIO_LIST_PROJECTION = {
    "audio_name": 1,
    "audio_name_en": 1,
    "audio_url": 1,
    "cover_url": 1,
    "description": 1,
    "short_description": 1,
    "short_description_en": 1,
    "vip": 1,
    "language": 1,
    "content_form_tags": 1,
    "acoustic_metrics": 1,
    "created_at": 1,
    "create_time": 1,
}


def _prescription_content_tags(values: list[str]) -> list[str]:
    resolved = resolve_audio_preferences(preferred=values)
    if resolved["sound_disabled"]:
        raise ValueError("unresolved_audio_preference")
    return resolved["preferred_codes"]


def _root_tag_query() -> dict[str, Any]:
    return {
        "type": _CONTENT_FORM,
        "status": _TAG_ENABLED,
        "$or": [
            {"parent_tag_id": {"$exists": False}},
            {"parent_tag_id": None},
            {"parent_tag_id": ""},
        ],
    }


def _paginate_docs(
    docs: list[dict[str, Any]],
    page: int | None,
    page_size: int | None,
    fetch_all: bool,
    settings: Settings,
) -> dict[str, Any]:
    total = len(docs)
    if fetch_all:
        if total > settings.fetch_all_hard_limit:
            raise InvalidAudioQueryError(f"全量条数超过上限 {settings.fetch_all_hard_limit}")
        return {"list": docs, "page": 1, "page_size": len(docs), "total": total}
    cur_page, size = _page_window(page, page_size, settings)
    start = (cur_page - 1) * size
    chunk = docs[start : start + size]
    return {"list": chunk, "page": cur_page, "page_size": size, "total": total}


def _page_window(
    page: int | None,
    page_size: int | None,
    settings: Settings,
) -> tuple[int, int]:
    cur_page = 1 if page is None else page
    size = settings.default_page_size if page_size is None else page_size
    if cur_page < 1 or size < 1:
        raise InvalidAudioQueryError("page / page_size 须 ≥ 1")
    return cur_page, min(size, settings.max_page_size)


def _map_tag_dict(doc: dict[str, Any]) -> dict[str, Any]:
    parent = doc.get("parent_tag_id")
    return {
        "type": str(doc.get("type") or ""),
        "code": str(doc.get("code") or ""),
        "name": str(doc.get("name") or ""),
        "name_en": str(doc.get("name_en") or ""),
        "id": str(doc.get("id") or doc.get("_id") or ""),
        "parent_tag_id": None if parent is None else str(parent),
        "status": str(doc.get("status") or ""),
    }


def _map_material(doc: dict[str, Any]) -> dict[str, Any]:
    """缓存/过滤用中间形态，保留 content_form_tags。"""
    return {
        "id": str(doc.get("id") or doc.get("_id") or ""),
        "audio_name": str(doc.get("audio_name") or ""),
        "audio_name_en": str(doc.get("audio_name_en") or ""),
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        "description": str(doc.get("description") or ""),
        "short_description": str(doc.get("short_description") or ""),
        "short_description_en": str(doc.get("short_description_en") or ""),
        "vip": _to_vip(doc.get("vip")),
        "language": str(doc.get("language") or ""),
        "content_form_tags": doc.get("content_form_tags") or [],
        "acoustic_metrics": _acoustic_metrics(doc.get("acoustic_metrics")),
        "created_at": doc.get("created_at") or doc.get("create_time") or "",
    }


def _doc_language(doc: dict[str, Any]) -> str:
    """归一化文档 language；缺失 / null / 空串 / 字面 None|null 视为无语言。"""
    raw = doc.get("language")
    if raw is None:
        return ""
    text = str(raw).strip()
    if text in {"", "None", "null", "NULL"}:
        return ""
    return text


def _filter_docs_by_request_language(
    docs: list[dict[str, Any]],
    language: str,
) -> list[dict[str, Any]]:
    """只保留 language 为空/null，或与请求 language 相同的文档（路径 A/B 共用）。"""
    wanted = resolve_search_language(language)
    return [
        doc
        for doc in docs
        if (doc_lang := _doc_language(doc)) == "" or doc_lang == wanted
    ]


def _to_audio_list_item(doc: dict[str, Any], *, language: str) -> dict[str, Any]:
    return {
        "id": str(doc.get("id") or ""),
        "audio_name": str(doc.get("audio_name") or ""),
        "audio_name_en": str(doc.get("audio_name_en") or ""),
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        # 量产列表展示用短描述，写入 description 出参（不单独返回 short_description）
        "description": str(doc.get("short_description") or ""),
        "description_en": str(doc.get("short_description_en") or ""),
        "vip": _to_vip(doc.get("vip")),
        "tag": _map_audio_tags(doc.get("content_form_tags"), language=language),
        "is_voice": _is_voice(doc.get("acoustic_metrics")),
    }


def _map_audio_tags(raw: Any, *, language: str) -> list[str]:
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        names.append(_tag_display_name(item, language))
    return names


def _tag_display_name(item: dict[str, Any], language: str) -> str:
    """zh 返回 name；en 优先 name_en，缺省回退 name。"""
    name = str(item.get("name") or "")
    name_en = str(item.get("name_en") or "")
    if language == "en":
        return name_en or name
    return name


def _acoustic_metrics(value: Any) -> dict[str, Any] | None:
    """只保留对象形态的声学指标；缺失不补字段。"""
    if isinstance(value, dict):
        return value
    return None


def _is_voice(metrics: Any) -> bool:
    """voice_presence_ratio 为 0、缺失或非数值时无人声；其余数值视为有人声。"""
    if not isinstance(metrics, dict):
        return False
    ratio = metrics.get("voice_presence_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        return False
    return ratio != 0


def _to_vip(value: Any) -> int:
    """库无 vip / 假值时返回 0；真值返回 1（兼容 bool/int/常见字符串）。"""
    if value is None or value is False:
        return 0
    if isinstance(value, (int, float)):
        return 1 if value != 0 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none", "null"}:
            return 0
        if normalized in {"1", "true", "yes", "on"}:
            return 1
        return 0
    return 1 if bool(value) else 0


def _tag_vector_for_language(tag: dict[str, Any], language: str) -> list[float] | None:
    """en 优先 vector_en，zh 优先 vector；缺省回退另一侧。"""
    primary = "vector_en" if language == "en" else "vector"
    fallback = "vector" if language == "en" else "vector_en"
    for key in (primary, fallback):
        value = tag.get(key)
        if isinstance(value, list) and value:
            return value
    return None


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() in ("", "None")


def _is_content_form_dict(tag: dict[str, Any]) -> bool:
    dimension = str(tag.get("dimension") or tag.get("type") or "")
    return dimension == _CONTENT_FORM


def _is_root_content_form_dict(tag: dict[str, Any]) -> bool:
    return _is_content_form_dict(tag) and _is_blank(tag.get("parent_tag_id"))


def _normalize_content_form_codes(codes: list[str] | None) -> set[str]:
    """问卷展示名或 code 能对上词典就留下。对不上的跳过，不让整次列表失败。"""
    resolved: set[str] = set()
    for code in codes or []:
        text = str(code).strip()
        if not text:
            continue
        try:
            canonical = normalize_tag_code(text)
        except ValueError:
            continue
        if canonical:
            resolved.add(canonical)
    return resolved


def _created_stamp(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _order_preferred_then_rest(
    docs: list[dict[str, Any]],
    preferred_ids: set[str],
) -> list[dict[str, Any]]:
    """偏好音频在前，其余在后。两段按创建时间从新到旧；时间相同保持原顺序。"""
    preferred = [doc for doc in docs if str(doc.get("id") or "") in preferred_ids]
    rest = [doc for doc in docs if str(doc.get("id") or "") not in preferred_ids]
    preferred.sort(key=lambda doc: _created_stamp(doc.get("created_at")), reverse=True)
    rest.sort(key=lambda doc: _created_stamp(doc.get("created_at")), reverse=True)
    return preferred + rest


def _han_words(words: list[str]) -> list[str]:
    return [word for word in words if any("\u4e00" <= char <= "\u9fff" for char in word)]


def _audio_ids_for_tag_ids(docs: list[dict[str, Any]], tag_ids: set[str]) -> set[str]:
    if not tag_ids:
        return set()
    found: set[str] = set()
    for doc in docs:
        audio_id = str(doc.get("id") or "")
        if audio_id and _has_content_form_tag_id(doc, tag_ids):
            found.add(audio_id)
    return found


def _audio_ids_by_literal(docs: list[dict[str, Any]], words: list[str]) -> set[str]:
    """词直接出现在音频名或标签名里。描述常写「未检测到雷声」这类否定，不做字面匹配。"""
    forms = {
        form
        for word in words
        for form in (word, word.removesuffix("声音"), word.removesuffix("声"))
        if len(form) >= 2
    }
    if not forms:
        return set()
    found: set[str] = set()
    for doc in docs:
        audio_id = str(doc.get("id") or "")
        names = [str(doc.get("audio_name") or "")]
        names.extend(
            str(item.get("name") or "")
            for item in doc.get("content_form_tags") or []
            if isinstance(item, dict)
        )
        if audio_id and any(form in name for form in forms for name in names):
            found.add(audio_id)
    return found


def _tag_scores(
    tags: list[dict[str, Any]],
    query: list[float],
    vector_field: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for tag in tags:
        vector = tag.get(vector_field)
        tag_id = str(tag.get("id") or "").strip()
        if tag_id and isinstance(vector, list) and vector:
            scores[tag_id] = _cosine(query, vector)
    return scores


def _tag_scores_for_language(
    tags: list[dict[str, Any]],
    query: list[float],
    language: str,
) -> dict[str, float]:
    """按请求 language 选标签名向量（en→vector_en，zh→vector，缺省回退）。"""
    scores: dict[str, float] = {}
    for tag in tags:
        vector = _tag_vector_for_language(tag, language)
        tag_id = str(tag.get("id") or "").strip()
        if tag_id and isinstance(vector, list) and vector:
            scores[tag_id] = _cosine(query, vector)
    return scores


def _tag_ids_near_top(
    scores: dict[str, float],
    threshold: float,
    margin: float | None,
) -> set[str]:
    """不低于 threshold 的标签；给了 margin 时还要与最高分相差不超过 margin。"""
    if not scores:
        return set()
    cutoff = threshold if margin is None else max(threshold, max(scores.values()) - margin)
    return {tag_id for tag_id, score in scores.items() if score >= cutoff}


def _spread_tag_ids(
    docs: list[dict[str, Any]],
    audio_ids: set[str],
    tags: list[dict[str, Any]],
    tag_scores: dict[str, float],
    min_score: float,
) -> set[str]:
    """描述命中的音频身上的二级标签，名字也和这个词相近的。一级标签（如自然声）不扩散。"""
    if not audio_ids:
        return set()
    roots = {str(tag.get("id") or "") for tag in tags if _is_blank(tag.get("parent_tag_id"))}
    carried = {
        str(item.get("tag_id") or "")
        for doc in docs
        if str(doc.get("id") or "") in audio_ids
        for item in doc.get("content_form_tags") or []
        if isinstance(item, dict)
    }
    return {
        tag_id
        for tag_id in carried
        if tag_id and tag_id not in roots and tag_scores.get(tag_id, 0.0) >= min_score
    }


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _move_disliked_to_end(
    docs: list[dict[str, Any]],
    disliked_ids: set[str],
) -> list[dict[str, Any]]:
    """讨厌的音频挪到末尾，并按创建时间从新到旧。其余音频保持原有顺序。"""
    if not disliked_ids:
        return docs
    kept = [doc for doc in docs if str(doc.get("id") or "") not in disliked_ids]
    disliked = [doc for doc in docs if str(doc.get("id") or "") in disliked_ids]
    disliked.sort(key=lambda doc: _created_stamp(doc.get("created_at")), reverse=True)
    return kept + disliked


def _has_content_form_code(doc: dict[str, Any], tag_code: str) -> bool:
    needle = tag_code.casefold()
    if not needle:
        return False
    for item in doc.get("content_form_tags") or []:
        if not isinstance(item, dict):
            continue
        if reference_tag(item) == needle:
            return True
    return False


def _has_content_form_tag_id(doc: dict[str, Any], tag_ids: set[str]) -> bool:
    if not tag_ids:
        return False
    for item in doc.get("content_form_tags") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("tag_id") or "") in tag_ids:
            return True
    return False

