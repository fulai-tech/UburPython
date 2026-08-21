"""手板 Mongo 原料集合访问（直连，非独立 mongo 包）。"""

from __future__ import annotations

from typing import Any

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import ReturnDocument

from app.core.bson_util import bson_to_jsonable, parse_object_id, utc_now
from app.core.config import Settings
from app.core.exceptions import MaterialNotFoundError

_CREATE_REQUIRED_DEFAULTS: dict[str, Any] = {
    "status": True,
    "audio_url": "",
    "operation_type": 0,
    "sleep_stage_tags": [],
    "content_form_tags": [],
    "mechanism_tags": [],
    "audio_engineering_tags": [],
    "medical_risk_tags": [],
    "evidence_level_tags": [],
}


class MaterialsStore:
    """somni_audio_materials 集合访问。"""

    def __init__(self, client: AsyncIOMotorClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    @property
    def _collection(self) -> AsyncIOMotorCollection:
        db = self._client[self._settings.mongo_db]
        return db[self._settings.mongo_materials_collection]

    async def insert_material(self, doc: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        payload = {**_CREATE_REQUIRED_DEFAULTS, **doc}
        payload["created_at"] = doc.get("created_at") or now
        payload["updated_at"] = now
        result = await self._collection.insert_one(payload)
        material_id = str(result.inserted_id)
        logger.info("Mongo 已创建原料，id={}", material_id)
        return self._as_response(payload, material_id)

    async def update_material(
        self,
        material_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        oid = parse_object_id(material_id)
        if not fields:
            return await self.get_material(material_id)
        payload = {**fields, "updated_at": utc_now()}
        doc = await self._collection.find_one_and_update(
            {"_id": oid},
            {"$set": payload},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            raise MaterialNotFoundError(material_id)
        logger.info("Mongo 已更新原料，id={}", material_id)
        return self._as_response(doc, material_id)

    async def delete_material(self, material_id: str) -> None:
        oid = parse_object_id(material_id)
        result = await self._collection.delete_one({"_id": oid})
        if result.deleted_count == 0:
            raise MaterialNotFoundError(material_id)
        logger.info("Mongo 已删除原料，id={}", material_id)

    async def get_material(self, material_id: str) -> dict[str, Any]:
        oid = parse_object_id(material_id)
        doc = await self._collection.find_one({"_id": oid})
        if doc is None:
            raise MaterialNotFoundError(material_id)
        return self._as_response(doc, material_id)

    def close(self) -> None:
        self._client.close()

    def _as_response(self, doc: dict[str, Any], material_id: str) -> dict[str, Any]:
        payload = bson_to_jsonable(doc)
        payload.pop("_id", None)
        payload["id"] = material_id
        return payload


def create_materials_store(settings: Settings) -> MaterialsStore | None:
    if not settings.mongo_uri:
        return None
    client = AsyncIOMotorClient(settings.mongo_uri)
    return MaterialsStore(client, settings)
