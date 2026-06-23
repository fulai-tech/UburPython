"""scripts/sync_es_from_comm.py Mongo 同步逻辑单元测试。"""

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
    bson_to_jsonable,
    material_doc_to_es,
    material_documents_differ,
    mongo_doc_id,
    start_sync_scheduler,
    tag_dictionary_compare_snapshot,
    tag_documents_differ,
    zero_vector,
)


def _material_doc(
    doc_id: str,
    *,
    audio_url: str = "https://cdn.example.com/a.mp3",
    audio_name: str = "测试音频",
) -> dict:
    return {
        "_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id,
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


def test_tag_documents_same_when_only_vectors_differ() -> None:
    desired = {"name": "放松", "name_en": "Unwind", "status": "启用"}
    existing = {
        "name": "放松",
        "name_en": "Unwind",
        "status": "启用",
        "name_vector": [0.1],
        "name_en_vector": [0.2],
    }
    assert tag_documents_differ(desired, existing) is False


def test_material_documents_differ_on_tag_change() -> None:
    desired = material_doc_to_es(_material_doc("6a33a7928030d4cf420efeb6"))
    existing = dict(desired)
    existing["sleep_stage_tags"] = []
    assert material_documents_differ(desired, existing) is True


@pytest.mark.asyncio
async def test_tag_sync_job_deletes_es_orphan() -> None:
    mongo = MagicMock()
    mongo.fetch_tag_dictionary = AsyncMock(return_value=[_tag_doc("6a325acc1a3dbc128504c423")])
    es_search = MagicMock()
    es_search.list_all_tag_dictionary_doc_ids = AsyncMock(
        return_value={"6a325acc1a3dbc128504c423", "orphan"}
    )
    es_search.get_tag_dictionary_source = AsyncMock(return_value=None)
    es_search.tag_dictionary_index = "somni_audio_tag_dictionary"
    es_client = MagicMock()
    es_client.index = AsyncMock()
    es_client.delete = AsyncMock()
    encoder = MagicMock()
    encoder.encode_one = AsyncMock(return_value=[0.1] * 512)

    stats = await TagDictionarySyncJob(
        mongo, es_search, es_client, encoder, Settings(sync_backup_dir="/tmp")
    ).run(dry_run=False)

    assert stats["deleted"] == 1
    es_client.delete.assert_awaited_once_with(
        index="somni_audio_tag_dictionary", id="orphan"
    )


@pytest.mark.asyncio
async def test_material_sync_job_skips_unchanged(tmp_path) -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6")
    es_payload = material_doc_to_es(doc)
    mongo = MagicMock()
    mongo.fetch_materials = AsyncMock(return_value=[doc])
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value={"6a33a7928030d4cf420efeb6"})
    es_search.get_audio_source = AsyncMock(return_value=es_payload)
    es_search.audio_index = "somni_audio_materials"
    es_client = MagicMock()
    es_client.index = AsyncMock()

    stats = await MaterialsSyncJob(
        mongo, es_search, es_client, Settings(sync_backup_dir=str(tmp_path))
    ).run(dry_run=False)

    assert stats["unchanged"] == 1
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
    es_client = MagicMock()
    encoder = MagicMock()

    await MongoEsSyncJob(mongo, es_search, es_client, encoder, Settings()).run(dry_run=True)

    es_search.migrate_legacy_indices.assert_awaited_once()
    es_search.ensure_indices.assert_awaited_once()


def test_zero_vector_has_embedding_dim_length() -> None:
    assert len(zero_vector(512)) == 512
    assert all(v == 0.0 for v in zero_vector(512))


def test_tag_dictionary_snapshot_uses_created_by_fields() -> None:
    doc = _tag_doc("6a325acc1a3dbc128504c423")
    snapshot = tag_dictionary_compare_snapshot(bson_to_jsonable(doc))
    assert snapshot["created_by"] == "tester"
    assert snapshot["updated_by"] == "tester"
    assert "create_by" not in snapshot


def test_start_sync_scheduler_skipped_without_mongo_uri() -> None:
    with patch("scripts.sync_es_from_comm.AsyncIOScheduler") as mock_cls:
        settings = Settings(sync_enabled=True, mongo_uri="")
        start_sync_scheduler(AppState(settings=settings), settings)
        mock_cls.assert_not_called()


def test_mongo_doc_id_from_object_id() -> None:
    oid = ObjectId("6a33a7928030d4cf420efeb6")
    assert mongo_doc_id({"_id": oid}) == "6a33a7928030d4cf420efeb6"
