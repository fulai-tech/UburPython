"""scripts/sync_es_from_comm.py Mongo 全量重建同步单元测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from app.core.config import Settings
from app.main import AppState
from scripts.sync_es_from_comm import (
    MaterialsSyncJob,
    MongoEsSyncJob,
    TagDictionarySyncJob,
    _redact_mongo_uri,
    bson_to_jsonable,
    material_doc_to_es,
    mongo_doc_id,
    start_sync_scheduler,
    tag_doc_to_es,
    wipe_and_recreate_index,
    zero_vector,
)


def test_redact_mongo_uri_hides_password() -> None:
    assert (
        _redact_mongo_uri("mongodb://Fullive:secret@18.167.165.48:27017/Fullive")
        == "mongodb://Fullive:***@18.167.165.48:27017/Fullive"
    )
    assert _redact_mongo_uri("mongodb://localhost:27017") == "mongodb://localhost:27017"


def _material_doc(
    doc_id: str,
    *,
    audio_url: str = "https://cdn.example.com/a.mp3",
    audio_name: str = "测试音频",
) -> dict:
    return {
        "_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id,
        "id": doc_id,
        "audio_name": audio_name,
        "description": "描述",
        "status": True,
        "audio_url": audio_url,
        "operation_type": 0,
        "created_by": "tester",
        "updated_by": "tester",
        "sleep_stage_tags": [{"tag_id": "t1", "code": "unwind", "name": "放松"}],
        "content_form_tags": [],
        "mechanism_tags": [],
        "audio_engineering_tags": [],
        "medical_risk_tags": [],
        "evidence_level_tags": [{"tag_id": "e1", "code": "B", "name": "中等证据"}],
        "created_at": datetime(2026, 6, 18, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 18, tzinfo=UTC),
    }


def _tag_doc(doc_id: str, *, name: str = "放松", name_en: str = "Unwind") -> dict:
    return {
        "_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id,
        "id": doc_id,
        "type": "sleep_stage",
        "code": "unwind",
        "status": "启用",
        "name": name,
        "name_en": name_en,
        "created_by": "tester",
        "updated_by": "tester",
        "created_at": datetime(2026, 6, 16, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 16, tzinfo=UTC),
    }


def test_bson_to_jsonable_converts_object_id_and_datetime() -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6")
    result = bson_to_jsonable(doc)
    assert result["_id"] == "6a33a7928030d4cf420efeb6"
    assert result["created_at"].endswith("Z")


def test_material_doc_to_es_requires_audio_url() -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6", audio_url="")
    assert material_doc_to_es(doc) is None


def test_material_doc_to_es_keeps_sleep_stage_names_without_id_fields() -> None:
    payload = material_doc_to_es(_material_doc("6a33a7928030d4cf420efeb6"))
    assert payload is not None
    assert "id" not in payload
    assert "_id" not in payload
    assert payload["sleep_stage_names"] == ["放松"]


def test_tag_doc_to_es_strips_id_fields() -> None:
    payload = tag_doc_to_es(_tag_doc("6a325acc1a3dbc128504c423"))
    assert payload is not None
    assert "id" not in payload
    assert "_id" not in payload
    assert payload["name"] == "放松"


@pytest.mark.asyncio
async def test_wipe_and_recreate_index_deletes_then_ensures() -> None:
    es_client = MagicMock()
    es_client.indices.exists = AsyncMock(return_value=True)
    es_client.count = AsyncMock(return_value={"count": 12})
    es_client.indices.delete = AsyncMock()
    es_search = MagicMock()
    es_search.ensure_indices = AsyncMock()

    deleted = await wipe_and_recreate_index(es_client, es_search, "somni_audio_materials")

    assert deleted == 12
    es_client.indices.delete.assert_awaited_once_with(index="somni_audio_materials")
    es_search.ensure_indices.assert_awaited_once()


@pytest.mark.asyncio
async def test_tag_sync_job_wipes_index_then_inserts() -> None:
    mongo = MagicMock()
    mongo.fetch_tag_dictionary = AsyncMock(return_value=[_tag_doc("6a325acc1a3dbc128504c423")])
    es_search = MagicMock()
    es_search.list_all_tag_dictionary_doc_ids = AsyncMock(return_value={"orphan"})
    es_search.tag_dictionary_index = "somni_audio_tag_dictionary"
    es_search.clear_content_tag_vectors_cache = MagicMock()
    es_search.ensure_indices = AsyncMock()
    es_client = MagicMock()
    es_client.indices.exists = AsyncMock(return_value=True)
    es_client.count = AsyncMock(return_value={"count": 1})
    es_client.indices.delete = AsyncMock()
    es_client.index = AsyncMock()
    encoder = MagicMock()
    encoder.encode_one = AsyncMock(return_value=[0.1] * 512)

    stats = await TagDictionarySyncJob(
        mongo, es_search, es_client, encoder, Settings(sync_backup_dir="/tmp")
    ).run(dry_run=False)

    assert stats["deleted"] == 1
    assert stats["created"] == 1
    es_client.indices.delete.assert_awaited_once_with(index="somni_audio_tag_dictionary")
    kwargs = es_client.index.await_args.kwargs
    assert kwargs["id"] == "6a325acc1a3dbc128504c423"
    assert "id" not in kwargs["document"]
    assert "_id" not in kwargs["document"]


@pytest.mark.asyncio
async def test_material_sync_job_wipes_then_reindexes(tmp_path) -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6")
    mongo = MagicMock()
    mongo.fetch_materials = AsyncMock(return_value=[doc])
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value={"6a33a7928030d4cf420efeb6"})
    es_search.audio_index = "somni_audio_materials"
    es_search.ensure_indices = AsyncMock()
    es_client = MagicMock()
    es_client.indices.exists = AsyncMock(return_value=True)
    es_client.count = AsyncMock(return_value={"count": 99})
    es_client.indices.delete = AsyncMock()
    es_client.index = AsyncMock()
    encoder = MagicMock()
    encoder.encode_one = AsyncMock(return_value=[0.1] * 512)

    stats = await MaterialsSyncJob(
        mongo, es_search, es_client, encoder, Settings(sync_backup_dir=str(tmp_path))
    ).run(dry_run=False)

    assert stats["deleted"] == 99
    assert stats["created"] == 1
    es_client.indices.delete.assert_awaited_once_with(index="somni_audio_materials")
    indexed = es_client.index.await_args.kwargs
    assert indexed["id"] == "6a33a7928030d4cf420efeb6"
    assert "id" not in indexed["document"]
    assert "_id" not in indexed["document"]
    assert indexed["document"]["sleep_stage_names"] == ["放松"]


@pytest.mark.asyncio
async def test_material_sync_job_writes_description_vector(tmp_path) -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6")
    mongo = MagicMock()
    mongo.fetch_materials = AsyncMock(return_value=[doc])
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value=set())
    es_search.audio_index = "somni_audio_materials"
    es_search.ensure_indices = AsyncMock()
    es_client = MagicMock()
    es_client.indices.exists = AsyncMock(return_value=False)
    es_client.indices.delete = AsyncMock()
    es_client.index = AsyncMock()
    encoder = MagicMock()
    encoder.encode_one = AsyncMock(return_value=[0.2] * 512)

    stats = await MaterialsSyncJob(
        mongo, es_search, es_client, encoder, Settings(sync_backup_dir=str(tmp_path))
    ).run(dry_run=False)

    assert stats["created"] == 1
    indexed = es_client.index.await_args.kwargs["document"]
    assert indexed["description_text"] == "测试音频 描述 放松 unwind 中等证据 B"
    assert indexed["description_vector"] == [0.2] * 512
    encoder.encode_one.assert_awaited_once_with(indexed["description_text"])


@pytest.mark.asyncio
async def test_material_sync_dry_run_does_not_wipe(tmp_path) -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6")
    mongo = MagicMock()
    mongo.fetch_materials = AsyncMock(return_value=[doc])
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value={"a", "b", "c"})
    es_search.audio_index = "somni_audio_materials"
    es_client = MagicMock()
    es_client.indices.delete = AsyncMock()
    es_client.index = AsyncMock()
    encoder = MagicMock()

    stats = await MaterialsSyncJob(
        mongo, es_search, es_client, encoder, Settings(sync_backup_dir=str(tmp_path))
    ).run(dry_run=True)

    assert stats["deleted"] == 3
    assert stats["created"] == 1
    es_client.indices.delete.assert_not_called()
    es_client.index.assert_not_called()


@pytest.mark.asyncio
async def test_mongo_sync_job_migrates_legacy_indices() -> None:
    mongo = MagicMock()
    mongo.fetch_tag_dictionary = AsyncMock(return_value=[])
    mongo.fetch_materials = AsyncMock(return_value=[])
    es_search = MagicMock()
    es_search.migrate_legacy_indices = AsyncMock()
    es_search.ensure_indices = AsyncMock()
    es_search.list_all_tag_dictionary_doc_ids = AsyncMock(return_value=set())
    es_search.list_all_audio_doc_ids = AsyncMock(return_value=set())
    es_search.tag_dictionary_index = "somni_audio_tag_dictionary"
    es_search.audio_index = "somni_audio_materials"
    es_search.clear_content_tag_vectors_cache = MagicMock()
    es_client = MagicMock()
    es_client.indices.exists = AsyncMock(return_value=False)
    encoder = MagicMock()

    await MongoEsSyncJob(mongo, es_search, es_client, encoder, Settings()).run(dry_run=True)

    es_search.migrate_legacy_indices.assert_awaited_once()
    es_search.ensure_indices.assert_awaited_once()


def test_zero_vector_has_embedding_dim_length() -> None:
    assert len(zero_vector(512)) == 512
    assert all(v == 0.0 for v in zero_vector(512))


def test_start_sync_scheduler_skipped_without_mongo_uri() -> None:
    with patch("scripts.sync_es_from_comm.AsyncIOScheduler") as mock_cls:
        settings = Settings(sync_enabled=True, mongo_uri="")
        start_sync_scheduler(AppState(settings=settings), settings)
        mock_cls.assert_not_called()


def test_mongo_doc_id_from_object_id() -> None:
    oid = ObjectId("6a33a7928030d4cf420efeb6")
    assert mongo_doc_id({"_id": oid}) == "6a33a7928030d4cf420efeb6"
