"""量产音频业务：标签 / 列表 / 文本搜索（直连 Somni Mongo + 量产 ES）。"""

from __future__ import annotations

import math
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.core.bson_util import bson_to_jsonable
from app.core.config import Settings
from app.core.exceptions import AppError, MongoNotConfiguredError
from app.core.codes import HttpStatus
from app.schemas.audio import SearchAudioData, SearchAudioRequest
from app.services.retrieval import RetrievalService


class InvalidListParamsError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=HttpStatus.BAD_REQUEST)


class SomniAudioService:
    def __init__(
        self,
        client: AsyncIOMotorClient | None,
        settings: Settings,
        retrieval: RetrievalService | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._retrieval = retrieval

    def _require_client(self) -> AsyncIOMotorClient:
        if self._client is None:
            raise MongoNotConfiguredError()
        return self._client

    def _materials(self) -> AsyncIOMotorCollection:
        client = self._require_client()
        db = client[self._settings.somni_mongo_db]
        return db[self._settings.somni_mongo_materials_collection]

    def _tags(self) -> AsyncIOMotorCollection:
        client = self._require_client()
        db = client[self._settings.somni_mongo_db]
        return db[self._settings.somni_mongo_tag_dictionary_collection]

    async def list_tags(
        self,
        *,
        page: int | None = None,
        page_size: int | None = None,
        fetch_all: bool = False,
        type_: str | None = None,
        enabled_only: bool = False,
        level: int = 0,
    ) -> dict[str, Any]:
        if level not in (0, 1, 2):
            raise InvalidListParamsError("level 仅支持 0 / 1 / 2")
        query: dict[str, Any] = {}
        if type_:
            query["type"] = type_
        if enabled_only:
            query["status"] = "启用"
        if level == 1:
            query["$or"] = [
                {"parent_tag_id": {"$exists": False}},
                {"parent_tag_id": None},
                {"parent_tag_id": ""},
            ]
        elif level == 2:
            query["parent_tag_id"] = {"$nin": [None, ""]}
        return await self._paginate(
            self._tags(),
            query,
            page=page,
            page_size=page_size,
            fetch_all=fetch_all,
            map_doc=self._map_tag,
            list_key="tags",
        )

    async def list_audios(
        self,
        *,
        page: int | None = None,
        page_size: int | None = None,
        fetch_all: bool = False,
        enabled_only: bool = False,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if enabled_only:
            query["status"] = True
        if tags:
            query["$and"] = [_tag_match_clause(t) for t in tags]
        return await self._paginate(
            self._materials(),
            query,
            page=page,
            page_size=page_size,
            fetch_all=fetch_all,
            map_doc=self._map_material,
            list_key="materials",
        )

    async def search_audio(self, query_text: str, top_k: int | None = None) -> SearchAudioData:
        text = query_text.strip()
        if not text:
            raise InvalidListParamsError("query_text 不能为空")
        if self._retrieval is None:
            raise AppError(
                message="检索服务未就绪",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        req = SearchAudioRequest(query_text=text, top_k=top_k)
        materials = await self._retrieval.search(req)
        return SearchAudioData(materials=materials)

    async def _paginate(
        self,
        collection: AsyncIOMotorCollection,
        query: dict[str, Any],
        *,
        page: int | None,
        page_size: int | None,
        fetch_all: bool,
        map_doc,
        list_key: str,
    ) -> dict[str, Any]:
        total = await collection.count_documents(query)
        settings = self._settings
        if fetch_all:
            if total > settings.fetch_all_hard_limit:
                raise InvalidListParamsError(
                    f"全量条数超过上限 {settings.fetch_all_hard_limit}"
                )
            cursor = collection.find(query)
            docs = [map_doc(bson_to_jsonable(d)) async for d in cursor]
            return {
                list_key: docs,
                "page": {
                    "page": 1,
                    "page_size": len(docs),
                    "total": total,
                    "total_pages": 1,
                },
            }
        cur_page = page or 1
        size = page_size or settings.default_page_size
        if cur_page < 1 or size < 1:
            raise InvalidListParamsError("page / page_size 须 ≥ 1")
        size = min(size, settings.max_page_size)
        skip = (cur_page - 1) * size
        cursor = collection.find(query).skip(skip).limit(size)
        docs = [map_doc(bson_to_jsonable(d)) async for d in cursor]
        total_pages = math.ceil(total / size) if size else 0
        return {
            list_key: docs,
            "page": {
                "page": cur_page,
                "page_size": size,
                "total": total,
                "total_pages": total_pages,
            },
        }

    def _map_tag(self, doc: dict[str, Any]) -> dict[str, Any]:
        name = str(doc.get("name") or "")
        name_en = str(doc.get("name_en") or "")
        display = name or name_en
        parent_id = str(doc.get("parent_tag_id") or "")
        return {
            "id": str(doc.get("id") or doc.get("_id") or ""),
            "display_name": display,
            "name": name,
            "name_en": name_en,
            "type": str(doc.get("type") or ""),
            "code": str(doc.get("code") or ""),
            "status": str(doc.get("status") or ""),
            "parent_tag_id": parent_id,
            "parent_tag_name": str(doc.get("parent_tag_name") or ""),
            "created_at": str(doc.get("created_at") or ""),
            "updated_at": str(doc.get("updated_at") or ""),
        }

    def _map_material(self, doc: dict[str, Any]) -> dict[str, Any]:
        mid = str(doc.get("id") or doc.get("_id") or "")
        return {
            "id": mid,
            "audio_name": str(doc.get("audio_name") or ""),
            "audio_url": str(doc.get("audio_url") or ""),
            "cover_url": str(doc.get("cover_url") or ""),
            "description": str(doc.get("description") or ""),
            "status": bool(doc.get("status", False)),
            "operation_type": int(doc.get("operation_type") or 0),
            "created_by": str(doc.get("created_by") or ""),
            "updated_by": str(doc.get("updated_by") or ""),
            "create_time": str(doc.get("create_time") or doc.get("created_at") or ""),
            "update_time": str(doc.get("update_time") or doc.get("updated_at") or ""),
            "sleep_stage_tags": doc.get("sleep_stage_tags") or [],
            "content_form_tags": doc.get("content_form_tags") or [],
            "mechanism_tags": doc.get("mechanism_tags") or [],
            "audio_engineering_tags": doc.get("audio_engineering_tags") or [],
            "medical_risk_tags": doc.get("medical_risk_tags") or [],
            "evidence_level_tags": doc.get("evidence_level_tags") or [],
        }


def _tag_match_clause(tag: str) -> dict[str, Any]:
    """素材各维标签 name/code 任一匹配。"""
    fields = (
        "sleep_stage_tags",
        "content_form_tags",
        "mechanism_tags",
        "audio_engineering_tags",
        "medical_risk_tags",
        "evidence_level_tags",
    )
    ors: list[dict[str, Any]] = []
    for field in fields:
        ors.append({f"{field}.name": tag})
        ors.append({f"{field}.code": tag})
    return {"$or": ors}
