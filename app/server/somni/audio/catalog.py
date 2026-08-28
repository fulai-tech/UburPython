"""量产音频目录查询：标签词典 + 音频原料。"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from time import monotonic
from typing import Any

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.core.bson_util import bson_to_jsonable
from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError, EncoderNotReadyError
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.server.somni.audio.hot import HotTracker

_TAG_ENABLED = "启用"
_CONTENT_FORM = "content_form"


def _normalize_tag_code(code: str) -> str:
    return code.strip().casefold()


@dataclass(frozen=True)
class _MatchedContentTags:
    ids: set[str]
    codes: set[str]


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
    ) -> None:
        self._client = client
        self._settings = settings
        self._es_search = es_search
        self._encoder = encoder
        self._hot = hot
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
    ) -> dict[str, Any]:
        text = query_text.strip()
        code = _normalize_tag_code(tag_code)
        docs = await self._load_audios(from_es=bool(text), language=language)
        if code:
            docs = [doc for doc in docs if _has_content_form_code(doc, code)]
        if text:
            matched = await self._content_form_tags_by_text(text, language=language)
            docs = [doc for doc in docs if _has_content_form_tag_id(doc, matched.ids)]
        payload = _paginate_docs(docs, page, page_size, fetch_all, self._settings)
        payload["list"] = [_to_audio_list_item(item) for item in payload["list"]]
        self._schedule_hot(query_text, int(payload.get("total") or 0))
        return payload

    async def get_hot(self) -> dict[str, Any]:
        if self._hot is None:
            raise AppError(
                message="量产 Redis 未配置，无法获取热点",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        return {"items": await self._hot.list_hot()}

    def _schedule_hot(self, query_text: str, hit_count: int) -> None:
        if self._hot is None or not query_text.strip():
            return
        task = asyncio.create_task(self._record_hot_safely(query_text, hit_count))
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

    async def _record_hot_safely(self, query_text: str, hit_count: int) -> None:
        async with self._hot_sem:
            try:
                await self._hot.record_search(query_text, hit_count=hit_count)
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
        return ttl > 0 and now - loaded_at < ttl

    async def _fetch_audios_mongo(self, language: str) -> list[dict[str, Any]]:
        collection = self._materials()
        query = {"language": language}
        total = await collection.count_documents(query)
        self._reject_over_limit(total)
        cursor = collection.find(query, {"embedding": 0})
        return [bson_to_jsonable(doc) async for doc in cursor]

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
        "vip": _to_vip(doc.get("vip")),
        "content_form_tags": doc.get("content_form_tags") or [],
    }


def _to_audio_list_item(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc.get("id") or ""),
        "audio_name": str(doc.get("audio_name") or ""),
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        "description": str(doc.get("description") or ""),
        "vip": _to_vip(doc.get("vip")),
    }


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


def _has_content_form_code(doc: dict[str, Any], tag_code: str) -> bool:
    needle = tag_code.casefold()
    if not needle:
        return False
    for item in doc.get("content_form_tags") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("code") or "").strip().casefold() == needle:
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
