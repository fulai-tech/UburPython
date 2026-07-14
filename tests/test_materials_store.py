"""MaterialsStore 时间字段须写 BSON date，避免 Mongo validator 拒绝。"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.core.config import Settings
from app.mongo.materials import MaterialsStore


def _store_with_collection() -> tuple[MaterialsStore, MagicMock]:
    settings = Settings(mongo_db="Fullive", mongo_materials_collection="somni_audio_materials")
    client = MagicMock()
    collection = client[settings.mongo_db][settings.mongo_materials_collection]
    return MaterialsStore(client, settings), collection


@pytest.mark.asyncio
async def test_update_material_sets_updated_at_as_datetime() -> None:
    material_id = "6a3a3d21884ab13242e2cc31"
    store, collection = _store_with_collection()
    collection.find_one_and_update = AsyncMock(
        return_value={
            "_id": ObjectId(material_id),
            "audio_name": "轻柔轻音乐伴雪花声2222222",
            "updated_at": datetime.now(),
        }
    )

    await store.update_material(material_id, {"audio_name": "轻柔轻音乐伴雪花声2222222"})

    update_doc = collection.find_one_and_update.await_args.args[1]
    updated_at = update_doc["$set"]["updated_at"]
    assert isinstance(updated_at, datetime)
    assert updated_at.tzinfo is not None


@pytest.mark.asyncio
async def test_insert_material_sets_timestamps_as_datetime() -> None:
    store, collection = _store_with_collection()
    inserted_id = ObjectId()
    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))

    await store.insert_material({"audio_name": "雨声"})

    payload = collection.insert_one.await_args.args[0]
    assert isinstance(payload["created_at"], datetime)
    assert isinstance(payload["updated_at"], datetime)
    assert payload["created_at"].tzinfo is not None
    assert payload["updated_at"].tzinfo is not None


@pytest.mark.asyncio
async def test_insert_material_fills_mongo_required_defaults() -> None:
    """仅 audio_name 时补齐 Mongo validator 必填字段。"""
    store, collection = _store_with_collection()
    inserted_id = ObjectId()
    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))

    await store.insert_material({"audio_name": "仅名称"})

    payload = collection.insert_one.await_args.args[0]
    assert payload["audio_name"] == "仅名称"
    assert payload["status"] is True
    assert payload["audio_url"] == ""
    assert payload["operation_type"] == 0
    for key in (
        "sleep_stage_tags",
        "content_form_tags",
        "mechanism_tags",
        "audio_engineering_tags",
        "medical_risk_tags",
        "evidence_level_tags",
    ):
        assert payload[key] == []


@pytest.mark.asyncio
async def test_insert_material_keeps_explicit_required_fields() -> None:
    store, collection = _store_with_collection()
    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))

    await store.insert_material(
        {
            "audio_name": "雨声",
            "status": False,
            "audio_url": "https://cdn.example.com/a.mp3",
            "operation_type": 1,
        }
    )

    payload = collection.insert_one.await_args.args[0]
    assert payload["status"] is False
    assert payload["audio_url"] == "https://cdn.example.com/a.mp3"
    assert payload["operation_type"] == 1
