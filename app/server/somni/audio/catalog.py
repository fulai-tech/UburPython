"""量产音频目录查询：标签词典 + 音频原料。"""

from __future__ import annotations

import asyncio
import math
from time import monotonic
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.core.bson_util import bson_to_jsonable
from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError, EncoderNotReadyError
from app.embedding.encoder import Encoder
from app.es.search import EsSearch

_TAG_ENABLED = "启用"
_CONTENT_FORM = "content_form"


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
    ) -> None:
        self._client = client
        self._settings = settings
        self._es_search = es_search
        self._encoder = encoder
        self._audio_cache: dict[bool, tuple[float, list[dict[str, Any]]]] = {}
        self._audio_cache_lock = asyncio.Lock()

    async def get_audio_tag(self) -> dict[str, Any]:
        collection = self._tags()
        query = _root_tag_query()
        total = await collection.count_documents(query)
        self._reject_over_limit(total)
        cursor = collection.find(query, {"type": 1, "code": 1, "name": 1, "name_en": 1})
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
    ) -> dict[str, Any]:
        text = query_text.strip()
        code = tag_code.strip()
        docs = await self._load_audios(from_es=bool(text))
        if code:
            docs = [doc for doc in docs if _has_content_form_code(doc, code)]
        if text:
            tag_ids = await self._root_tag_ids_by_text(text)
            docs = [doc for doc in docs if _has_root_content_form_id(doc, tag_ids)]
        return _paginate_docs(docs, page, page_size, fetch_all, self._settings)

    async def get_hot(self) -> None:
        return None

    async def _load_audios(self, *, from_es: bool) -> list[dict[str, Any]]:
        now = monotonic()
        cached = self._audio_cache.get(from_es)
        if cached is not None and self._is_cache_fresh(cached[0], now):
            return cached[1]
        async with self._audio_cache_lock:
            cached = self._audio_cache.get(from_es)
            if cached is not None and self._is_cache_fresh(cached[0], now):
                return cached[1]
            raw = await self._fetch_audios_es() if from_es else await self._fetch_audios_mongo()
            docs = [_map_material(doc) for doc in raw]
            self._audio_cache[from_es] = (now, docs)
            return docs

    def _is_cache_fresh(self, loaded_at: float, now: float) -> bool:
        ttl = self._settings.somni_audio_catalog_cache_ttl_sec
        return ttl > 0 and now - loaded_at < ttl

    async def _fetch_audios_mongo(self) -> list[dict[str, Any]]:
        collection = self._materials()
        total = await collection.count_documents({})
        self._reject_over_limit(total)
        cursor = collection.find({}, {"embedding": 0})
        return [bson_to_jsonable(doc) async for doc in cursor]

    async def _fetch_audios_es(self) -> list[dict[str, Any]]:
        if self._es_search is None:
            raise AppError(
                message="Elasticsearch 未就绪，无法按搜索词查询音频",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        docs = await self._es_search.list_audio_catalog_docs(
            size=self._settings.fetch_all_hard_limit + 1,
        )
        self._reject_over_limit(len(docs))
        return docs

    async def _root_tag_ids_by_text(self, text: str) -> set[str]:
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
        matched: set[str] = set()
        for tag in tags:
            if not _is_root_content_form_dict(tag):
                continue
            vector = tag.get("vector")
            if not isinstance(vector, list) or not vector:
                continue
            if _cosine_similarity(query_vector, vector) <= threshold:
                continue
            tag_id = str(tag.get("id") or "").strip()
            if tag_id:
                matched.add(tag_id)
        return matched

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


def _root_tag_query() -> dict[str, Any]:
    return {
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
        return {"materials": docs, "page": _page_info(1, len(docs), total, 1)}
    cur_page, size = _page_window(page, page_size, settings)
    start = (cur_page - 1) * size
    chunk = docs[start : start + size]
    pages = math.ceil(total / size) if size else 0
    return {"materials": chunk, "page": _page_info(cur_page, size, total, pages)}


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


def _page_info(page: int, page_size: int, total: int, total_pages: int) -> dict[str, int]:
    return {"page": page, "page_size": page_size, "total": total, "total_pages": total_pages}


def _map_tag_dict(doc: dict[str, Any]) -> dict[str, str]:
    return {
        "type": str(doc.get("type") or ""),
        "code": str(doc.get("code") or ""),
        "name": str(doc.get("name") or ""),
        "name_en": str(doc.get("name_en") or ""),
    }


def _map_material(doc: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(doc)
    mapped.pop("embedding", None)
    mapped["id"] = str(doc.get("id") or doc.get("_id") or "")
    mapped.pop("_id", None)
    return mapped


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() in ("", "None")


def _is_root_content_form_dict(tag: dict[str, Any]) -> bool:
    dimension = str(tag.get("dimension") or tag.get("type") or "")
    return dimension == _CONTENT_FORM and _is_blank(tag.get("parent_tag_id"))


def _has_content_form_code(doc: dict[str, Any], tag_code: str) -> bool:
    for item in doc.get("content_form_tags") or []:
        if isinstance(item, dict) and str(item.get("code") or "") == tag_code:
            return True
    return False


def _has_root_content_form_id(doc: dict[str, Any], tag_ids: set[str]) -> bool:
    if not tag_ids:
        return False
    for item in doc.get("content_form_tags") or []:
        if not isinstance(item, dict) or not _is_blank(item.get("parent_tag_id")):
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
