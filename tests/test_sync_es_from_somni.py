"""scripts/sync_es_from_somni.py 量产 Mongo → ES 全量重建同步单元测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from app.core.config import Settings
from scripts.sync_es_from_somni import (
    MaterialsSyncJob,
    MongoEsSyncJob,
    MongoSource,
    TagDictionarySyncJob,
    _redact_mongo_uri,
    material_doc_to_es,
    mongo_doc_id,
    wipe_and_recreate_index,
)


def test_redact_mongo_uri_hides_password() -> None:
    assert (
        _redact_mongo_uri("mongodb://Somni:secret@18.167.165.48:27017/Somni")
        == "mongodb://Somni:***@18.167.165.48:27017/Somni"
    )


def test_somni_sync_backup_paths_differ_from_handboard() -> None:
    settings = Settings(sync_backup_dir="/tmp/backups")
    assert settings.somni_sync_backup_path.name == "somni_prod_audio_materials_backup.json"
    assert (
        settings.somni_sync_tag_dictionary_backup_path.name
        == "somni_prod_audio_tag_dictionary_backup.json"
    )
    assert settings.somni_sync_backup_path != settings.sync_backup_path
    assert (
        settings.somni_sync_tag_dictionary_backup_path
        != settings.sync_tag_dictionary_backup_path
    )


def test_mongo_source_requires_somni_mongo_uri() -> None:
    with pytest.raises(ValueError, match="SOMNI_MONGO_URI"):
        MongoSource(Settings(somni_mongo_uri="", mongo_uri="mongodb://handboard"))


def test_mongo_source_uses_somni_db_and_collections() -> None:
    settings = Settings(
        somni_mongo_uri="mongodb://localhost:27017",
        somni_mongo_db="SomniProd",
        somni_mongo_materials_collection="prod_materials",
        somni_mongo_tag_dictionary_collection="prod_tags",
        mongo_uri="mongodb://handboard",
        mongo_db="Fullive",
    )
    with patch("scripts.sync_es_from_somni.AsyncIOMotorClient") as mock_client:
        db = MagicMock()
        mock_client.return_value.__getitem__.return_value = db
        source = MongoSource(settings)
    mock_client.assert_called_once_with("mongodb://localhost:27017")
    mock_client.return_value.__getitem__.assert_called_once_with("SomniProd")
    assert source._materials == "prod_materials"
    assert source._dictionary == "prod_tags"


def _material_doc(
    doc_id: str,
    *,
    audio_url: str = "https://cdn.example.com/a.mp3",
    language: str = "zh",
) -> dict:
    return {
        "_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id,
        "id": doc_id,
        "audio_name": "测试音频",
        "description": "描述",
        "status": True,
        "language": language,
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


def _tag_doc(doc_id: str) -> dict:
    return {
        "_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id,
        "id": doc_id,
        "type": "sleep_stage",
        "code": "unwind",
        "status": "启用",
        "name": "放松",
        "name_en": "Unwind",
        "created_by": "tester",
        "updated_by": "tester",
        "created_at": datetime(2026, 6, 16, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 16, tzinfo=UTC),
    }


def test_material_doc_to_es_requires_audio_url() -> None:
    assert material_doc_to_es(_material_doc("6a33a7928030d4cf420efeb6", audio_url="")) is None


def test_material_doc_to_es_keeps_language() -> None:
    payload = material_doc_to_es(
        _material_doc("6a33a7928030d4cf420efeb6", language="en")
    )
    assert payload is not None
    assert payload["language"] == "en"
    assert "id" not in payload
    assert "_id" not in payload


def test_material_doc_to_es_drops_embedding() -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6")
    doc["embedding"] = []
    payload = material_doc_to_es(doc)
    assert payload is not None
    assert "embedding" not in payload


def test_mongo_doc_id_from_object_id() -> None:
    oid = ObjectId("6a33a7928030d4cf420efeb6")
    assert mongo_doc_id({"_id": oid}) == "6a33a7928030d4cf420efeb6"


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
async def test_tag_sync_job_writes_somni_backup(tmp_path) -> None:
    mongo = MagicMock()
    mongo.fetch_tag_dictionary = AsyncMock(return_value=[_tag_doc("6a325acc1a3dbc128504c423")])
    es_search = MagicMock()
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
        mongo,
        es_search,
        es_client,
        encoder,
        Settings(sync_backup_dir=str(tmp_path)),
    ).run(dry_run=False)

    assert stats["created"] == 1
    backup = tmp_path / "somni_prod_audio_tag_dictionary_backup.json"
    assert backup.is_file()
    assert not (tmp_path / "somni_audio_tag_dictionary_backup.json").exists()


@pytest.mark.asyncio
async def test_material_sync_job_wipes_then_reindexes(tmp_path) -> None:
    doc = _material_doc("6a33a7928030d4cf420efeb6")
    mongo = MagicMock()
    mongo.fetch_materials = AsyncMock(return_value=[doc])
    es_search = MagicMock()
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
        mongo,
        es_search,
        es_client,
        encoder,
        Settings(sync_backup_dir=str(tmp_path)),
    ).run(dry_run=False)

    assert stats["deleted"] == 99
    assert stats["created"] == 1
    backup = tmp_path / "somni_prod_audio_materials_backup.json"
    assert backup.is_file()
    indexed = es_client.index.await_args.kwargs["document"]
    assert indexed["language"] == "zh"


@pytest.mark.asyncio
async def test_material_sync_dry_run_does_not_wipe(tmp_path) -> None:
    mongo = MagicMock()
    mongo.fetch_materials = AsyncMock(
        return_value=[_material_doc("6a33a7928030d4cf420efeb6")]
    )
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value={"a", "b", "c"})
    es_search.audio_index = "somni_audio_materials"
    es_client = MagicMock()
    es_client.indices.delete = AsyncMock()
    es_client.index = AsyncMock()

    stats = await MaterialsSyncJob(
        mongo,
        es_search,
        es_client,
        MagicMock(),
        Settings(sync_backup_dir=str(tmp_path)),
    ).run(dry_run=True)

    assert stats["deleted"] == 3
    assert stats["created"] == 1
    es_client.indices.delete.assert_not_called()
    es_client.index.assert_not_called()
    assert not (tmp_path / "somni_prod_audio_materials_backup.json").exists()


@pytest.mark.asyncio
async def test_material_sync_skips_missing_audio_url(tmp_path) -> None:
    mongo = MagicMock()
    mongo.fetch_materials = AsyncMock(
        return_value=[_material_doc("6a33a7928030d4cf420efeb6", audio_url="")]
    )
    es_search = MagicMock()
    es_search.audio_index = "somni_audio_materials"
    es_search.ensure_indices = AsyncMock()
    es_client = MagicMock()
    es_client.indices.exists = AsyncMock(return_value=False)
    es_client.index = AsyncMock()

    stats = await MaterialsSyncJob(
        mongo,
        es_search,
        es_client,
        MagicMock(),
        Settings(sync_backup_dir=str(tmp_path)),
    ).run(dry_run=False)

    assert stats["skipped"] == 1
    assert stats["created"] == 0
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

    await MongoEsSyncJob(
        mongo, es_search, es_client, MagicMock(), Settings()
    ).run(dry_run=True)

    es_search.migrate_legacy_indices.assert_awaited_once()
    es_search.ensure_indices.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_cli_uses_somni_es_node_and_indexes() -> None:
    settings = Settings(
        somni_mongo_uri="mongodb://localhost:27017",
        somni_es_node="http://somni-es:9200",
        es_node="http://handboard-es:9200",
        somni_es_audio_index="prod_audio",
        somni_es_tag_vectors_index="prod_tags",
        embedding_backend="torch",
    )
    es_client = MagicMock()
    es_client.close = AsyncMock()
    encoder = MagicMock()
    encoder.load = MagicMock()
    mongo = MagicMock()
    mongo.close = AsyncMock()
    job = MagicMock()
    job.run = AsyncMock(
        return_value=MagicMock(failed=0, material_created=0, material_updated=0, material_deleted=0)
    )

    with (
        patch("scripts.sync_es_from_somni.get_settings", return_value=settings),
        patch("scripts.sync_es_from_somni.setup_logging"),
        patch("scripts.sync_es_from_somni.create_es_client", return_value=es_client) as mk_es,
        patch("scripts.sync_es_from_somni.create_encoder", return_value=encoder),
        patch("scripts.sync_es_from_somni.MongoSource", return_value=mongo),
        patch("scripts.sync_es_from_somni.EsSearch") as mk_search,
        patch("scripts.sync_es_from_somni.MongoEsSyncJob", return_value=job),
    ):
        from scripts.sync_es_from_somni import _run_cli

        code = await _run_cli(dry_run=True)

    assert code == 0
    mk_es.assert_called_once_with(settings, node="http://somni-es:9200")
    mk_search.assert_called_once()
    kwargs = mk_search.call_args.kwargs
    assert kwargs["audio_index"] == "prod_audio"
    assert kwargs["tag_dictionary_index"] == "prod_tags"
