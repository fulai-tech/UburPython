"""量产音频目录查询：标签词典 + 音频原料。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
from typing import Any

from bson import ObjectId
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.core.audio_tags import (
    canonical_disliked_tags,
    canonical_tags,
    material_is_excluded,
    material_tag_codes,
    reference_tag,
    resolve_audio_preferences,
    validated_audio_tags,
)
from app.core.bson_util import bson_to_jsonable
from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError, EncoderNotReadyError
from app.embedding.encoder import Encoder
from app.es.search import CONTENT_TAG_TYPES, EsSearch
from app.schemas.search_material import project_search_material
from app.server.somni.audio.hot import HotTracker, normalize_tag_code
from app.server.somni.audio.segment import segment_dislike_text, segment_dislike_text_en

_TAG_ENABLED = "启用"
_CONTENT_FORM = "content_form"


@dataclass(frozen=True)
class _MatchedContentTags:
    ids: set[str]
    codes: set[str]


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
        encoder_en: Encoder | None = None,
        hot: HotTracker | None = None,
        retrieval: Any | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._es_search = es_search
        self._encoder = encoder
        self._encoder_en = encoder_en
        self._hot = hot
        self._content_form_tag_bank_cache: list[dict[str, Any]] | None = None
        self._retrieval = retrieval
        self._audio_cache: dict[tuple[bool, str], tuple[float, list[dict[str, Any]]]] = {}
        self._audio_cache_lock = asyncio.Lock()
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
        # 搜索词走原检索，不套问卷偏好。分类筛选仍保留偏好排序。
        if text:
            codes = []
            pref = ""
            dislike = ""
        docs = await self._load_audios(from_es=bool(text), language=language)
        hot_tag_codes: set[str] = set()
        if code:
            docs = [doc for doc in docs if _has_content_form_code(doc, code)]
            hot_tag_codes.add(code)
        if text:
            matched = await self._content_form_tags_by_text(text, language=language)
            docs = [doc for doc in docs if _has_content_form_tag_id(doc, matched.ids)]
            hot_tag_codes |= matched.codes
        preferred_ids: set[str] = set()
        if codes or pref:
            preferred_ids = await self._preferred_audio_ids(
                docs,
                codes=codes,
                preference_text=pref,
            )
            docs = _order_preferred_then_rest(docs, preferred_ids)
        docs = _move_disliked_to_end(docs, await self._disliked_audio_ids(docs, dislike))
        payload = _paginate_docs(docs, page, page_size, fetch_all, self._settings)
        payload["list"] = [_to_audio_list_item(item, language=language) for item in payload["list"]]
        self._schedule_hot(
            query_text,
            language,
            int(payload.get("total") or 0),
            tag_codes=hot_tag_codes,
        )
        return payload

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

    async def _disliked_audio_ids(self, docs: list[dict[str, Any]], dislike_text: str) -> set[str]:
        """讨厌的自由输入同样分词后对标签名向量，命中的音频排到最后。"""
        if not dislike_text.strip():
            return set()
        return _audio_ids_for_tag_ids(docs, await self._free_text_tag_ids(dislike_text))

    async def _free_text_tag_ids(self, text: str) -> set[str]:
        """每个词单独编码。中文对标签中文名向量，英文对英文模型现算的英文名向量。"""
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
                await self._tag_ids_for_words(tags, han_words, self._encoder, "vector", threshold)
            )
        if en_words:
            await self._fill_english_name_vectors(tags)
            matched.update(
                await self._tag_ids_for_words(
                    tags,
                    en_words,
                    self._encoder_en,
                    "vector_en_model",
                    threshold,
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
    ) -> set[str]:
        if encoder is None or not encoder.is_loaded:
            logger.warning(
                "自由输入未做{}向量：编码器未就绪", "英文" if vector_field != "vector" else "中文"
            )
            return set()
        matched: set[str] = set()
        for word in words:
            try:
                query = await encoder.encode_one(word)
            except Exception as exc:  # noqa: BLE001
                logger.warning("自由输入词向量失败：{}", exc)
                continue
            matched.update(_tag_ids_at_least(tags, query, vector_field, threshold))
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

    async def _fill_english_name_vectors(self, tags: list[dict[str, Any]]) -> None:
        """库里的 name_en_vector 是中文模型的 512 维，不能和英文模型比。按英文名现算 384 维。"""
        encoder = self._encoder_en
        if encoder is None or not encoder.is_loaded:
            return
        for tag in tags:
            if "vector_en_model" in tag:
                continue
            name_en = str(tag.get("name_en") or "").strip()
            if not name_en:
                tag["vector_en_model"] = []
                continue
            try:
                tag["vector_en_model"] = await encoder.encode_one(name_en)
            except Exception as exc:  # noqa: BLE001
                logger.warning("英文标签名向量失败：{}", exc)
                tag["vector_en_model"] = []

    async def search_prescription_audio(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Use the configured ranked retrieval service without recording hot queries."""
        if self._retrieval is None:
            raise AppError(message="检索服务未就绪", status_code=HttpStatus.SERVICE_UNAVAILABLE)
        from app.schemas.audio import SearchAudioRequest

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
                            else frozenset({"content_form", "mechanism", "audio_engineering"})
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
                    "type": {"$in": list(CONTENT_TAG_TYPES)},
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
                "mechanism_tags",
                "audio_engineering_tags",
                "medical_risk_tags",
                "evidence_level_tags",
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
            "medical_risk_tags",
            "mechanism_tags",
        )
        for candidate in candidates:
            identifier = str(candidate.get("_id") or candidate.get("id") or "")
            source = current.get(identifier)
            if source is None or source.get("status") is not True or not source.get("audio_url"):
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

    async def _load_audios(self, *, from_es: bool, language: str) -> list[dict[str, Any]]:
        cache_key = (from_es, language)
        now = monotonic()
        cached = self._audio_cache.get(cache_key)
        if cached is not None and self._is_cache_fresh(cached[0], now):
            return cached[1]
        async with self._audio_cache_lock:
            cached = self._audio_cache.get(cache_key)
            if cached is not None and self._is_cache_fresh(cached[0], now):
                return cached[1]
            raw = (
                await self._fetch_audios_es(language)
                if from_es
                else await self._fetch_audios_mongo(language)
            )
            docs = [_map_material(doc) for doc in raw]
            self._audio_cache[cache_key] = (now, docs)
            return docs

    def _is_cache_fresh(self, loaded_at: float, now: float) -> bool:
        ttl = self._settings.somni_audio_catalog_cache_ttl_sec
        if ttl <= 0:
            return True
        return now - loaded_at < ttl

    async def _fetch_audios_mongo(self, language: str) -> list[dict[str, Any]]:
        collection = self._materials()
        query = {"language": language}
        limit = self._settings.fetch_all_hard_limit + 1
        cursor = collection.find(query, _AUDIO_LIST_PROJECTION, limit=limit)
        docs = [bson_to_jsonable(doc) async for doc in cursor]
        self._reject_over_limit(len(docs))
        return docs

    async def _fetch_audios_es(self, language: str) -> list[dict[str, Any]]:
        if self._es_search is None:
            raise AppError(
                message="Elasticsearch 未就绪，无法按搜索词查询音频",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        docs = await self._es_search.list_audio_catalog_docs(
            size=self._settings.fetch_all_hard_limit + 1,
            language=language,
        )
        self._reject_over_limit(len(docs))
        return docs

    async def _content_form_tags_by_text(
        self,
        text: str,
        *,
        language: str,
    ) -> _MatchedContentTags:
        if self._encoder is None or not self._encoder.is_loaded:
            raise EncoderNotReadyError()
        if self._es_search is None:
            raise AppError(
                message="Elasticsearch 未就绪，无法按搜索词匹配标签",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        query_vector = await self._encoder.encode_one(text)
        tags = await self._es_search.list_content_tag_vectors()
        threshold = self._settings.get_audio_root_tag_sim_threshold
        scored: list[tuple[float, str]] = []
        lexical_ids: set[str] = set()
        id_to_code: dict[str, str] = {}
        for tag in tags:
            if not _is_content_form_dict(tag):
                continue
            tag_id = str(tag.get("id") or "").strip()
            if not tag_id:
                continue
            code = str(tag.get("code") or "").strip()
            if code:
                id_to_code[tag_id] = code
            if _lexical_match_content_tag(tag, text):
                lexical_ids.add(tag_id)
            vector = _tag_vector_for_language(tag, language)
            if not isinstance(vector, list) or not vector:
                continue
            sim = _cosine_similarity(query_vector, vector)
            if sim <= threshold:
                continue
            scored.append((sim, tag_id))
        matched_ids = lexical_ids | _select_matched_tag_ids(scored)
        return _MatchedContentTags(
            ids=matched_ids,
            codes={id_to_code[i] for i in matched_ids if i in id_to_code},
        )

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


def _select_matched_tag_ids(scored: list[tuple[float, str]]) -> set[str]:
    """保留高分标签；近精确命中时收紧范围，避免宽泛根标签稀释结果。"""
    if not scored:
        return set()
    best = max(sim for sim, _ in scored)
    if best >= 0.9:
        return {tag_id for sim, tag_id in scored if sim >= best - 0.05}
    return {tag_id for _, tag_id in scored}


_AUDIO_LIST_PROJECTION = {
    "audio_name": 1,
    "audio_url": 1,
    "cover_url": 1,
    "description": 1,
    "short_description": 1,
    "vip": 1,
    "content_form_tags": 1,
    "created_at": 1,
    "create_time": 1,
}


def _prescription_content_tags(values: list[str]) -> list[str]:
    resolved = resolve_audio_preferences(preferred=values)
    if resolved["unresolved_preferred"] or resolved["sound_disabled"]:
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
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        "description": str(doc.get("description") or ""),
        "short_description": str(doc.get("short_description") or ""),
        "vip": _to_vip(doc.get("vip")),
        "content_form_tags": doc.get("content_form_tags") or [],
        "created_at": doc.get("created_at") or doc.get("create_time") or "",
    }


def _to_audio_list_item(doc: dict[str, Any], *, language: str) -> dict[str, Any]:
    return {
        "id": str(doc.get("id") or ""),
        "audio_name": str(doc.get("audio_name") or ""),
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        # 量产列表展示用短描述，写入 description 出参（不单独返回 short_description）
        "description": str(doc.get("short_description") or ""),
        "vip": _to_vip(doc.get("vip")),
        "tag": _map_audio_tags(doc.get("content_form_tags"), language=language),
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
    """en 优先 name_en_vector，zh 优先 name_vector；缺省回退另一侧。"""
    primary = "vector_en" if language == "en" else "vector"
    fallback = "vector" if language == "en" else "vector_en"
    for key in (primary, fallback):
        value = tag.get(key)
        if isinstance(value, list) and value:
            return value
    return None


def _lexical_match_content_tag(tag: dict[str, Any], text: str) -> bool:
    """短英文词（如 rain）靠向量难过阈值时，用 code / 英文名词法命中。"""
    needle = text.strip().casefold()
    if not needle:
        return False
    code = str(tag.get("code") or "").strip().casefold()
    name = str(tag.get("label") or "").strip().casefold()
    name_en = str(tag.get("name_en") or "").strip().casefold()
    if needle in {code, name, name_en}:
        return True
    if code and needle in code.split("_"):
        return True
    if name_en and needle in name_en.replace("/", " ").split():
        return True
    return False


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() in ("", "None")


def _is_content_form_dict(tag: dict[str, Any]) -> bool:
    dimension = str(tag.get("dimension") or tag.get("type") or "")
    return dimension == _CONTENT_FORM


def _is_root_content_form_dict(tag: dict[str, Any]) -> bool:
    return _is_content_form_dict(tag) and _is_blank(tag.get("parent_tag_id"))


def _normalize_content_form_codes(codes: list[str] | None) -> set[str]:
    """问卷旧展示别名及新 code 使用同一英文标签身份。"""
    return {normalize_tag_code(str(code)) for code in (codes or []) if str(code).strip()}


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


def _tag_ids_at_least(
    tags: list[dict[str, Any]],
    query: list[float],
    vector_field: str,
    threshold: float,
) -> set[str]:
    matched: set[str] = set()
    for tag in tags:
        vector = tag.get(vector_field)
        if not isinstance(vector, list) or not vector:
            continue
        if _cosine(query, vector) >= threshold:
            tag_id = str(tag.get("id") or "").strip()
            if tag_id:
                matched.add(tag_id)
    return matched


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


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(y * y for y in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)
