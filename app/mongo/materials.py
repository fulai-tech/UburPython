"""Mongo somni_audio_materials 读写。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import ReturnDocument

from app.core.config import Settings
from app.core.exceptions import MaterialNotFoundError

# Mongo $jsonSchema required；HTTP 侧仅 audio_name 必填时由此补齐
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


def utc_now() -> datetime:
    """Mongo 校验要求 created_at / updated_at 为 BSON date。"""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def bson_to_jsonable(value: Any) -> Any:
    """BSON → JSON 可序列化（HTTP / ES）。"""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: bson_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bson_to_jsonable(v) for v in value]
    return value


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
        """插入原料，返回含 id 的 JSON 文档。"""
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
        """按字段 `$set` 更新；fields 为空则直接返回当前文档。"""
        oid = self._parse_object_id(material_id)
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

    async def get_material(self, material_id: str) -> dict[str, Any]:
        oid = self._parse_object_id(material_id)
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

    @staticmethod
    def _parse_object_id(material_id: str) -> ObjectId:
        try:
            return ObjectId(material_id)
        except InvalidId as exc:
            raise MaterialNotFoundError(material_id) from exc


def create_materials_store(settings: Settings) -> MaterialsStore | None:
    """有 mongo_uri 时创建 store，否则返回 None。"""
    if not settings.mongo_uri:
        return None
    client = AsyncIOMotorClient(settings.mongo_uri)
    return MaterialsStore(client, settings)
