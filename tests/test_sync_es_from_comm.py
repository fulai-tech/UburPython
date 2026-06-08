"""scripts/sync_es_from_comm.py 单文件同步逻辑单元测试。"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import Settings
from app.main import AppState
from scripts.sync_es_from_comm import (
    AudioEsSyncJob,
    audio_documents_differ,
    build_compare_snapshot,
    build_payloads_from_materials,
    compare_snapshot_from_es_source,
    content_tags_to_flat,
    material_to_es_payload,
    start_sync_scheduler,
)


@dataclass
class _FakeMetaData:
    url: str = ""


@dataclass
class _FakeAudioInfo:
    meta_data: _FakeMetaData = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta_data is None:
            self.meta_data = _FakeMetaData()


@dataclass
class _FakeMaterial:
    id: str = "69dcf1cba9c5f1466e6cccc0"
    category_code: int = 12
    name: str = "海浪声白噪音"
    tags: list[str] | None = None
    audio_info: _FakeAudioInfo | None = None

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = ["助眠", "放松", "白噪音"]
        if self.audio_info is None:
            self.audio_info = _FakeAudioInfo(
                meta_data=_FakeMetaData(
                    url="https://cdn.fulai.tech/comm/audio/1776085236_FjjxVcSD320.mp3",
                )
            )


def _make_material(
    material_id: str,
    *,
    url: str = "https://cdn.example.com/a.mp3",
    tags: list[str] | None = None,
) -> _FakeMaterial:
    return _FakeMaterial(
        id=material_id,
        name=f"音频-{material_id}",
        tags=tags or ["白噪音"],
        audio_info=_FakeAudioInfo(meta_data=_FakeMetaData(url=url)),
    )


def _existing_es_doc() -> dict:
    return {
        "audio_url": "https://cdn.example.com/a.mp3",
        "audio_name": "音频-id1",
        "evidence_level": "C",
        "recommend_weight": 0.45,
        "tags": {
            "sleep_stage": [],
            "content_form": [{"vector_id": "v2", "label": "白噪音"}],
            "mechanism": [],
            "audio_feat": [],
            "rhythm": [],
            "risk_control": [],
        },
    }


def test_content_tags_to_flat_adds_prefix() -> None:
    assert content_tags_to_flat(["助眠", "白噪音"]) == ["content:助眠", "content:白噪音"]


def test_material_to_es_payload_maps_example_document() -> None:
    payload = material_to_es_payload(_FakeMaterial())
    assert payload is not None
    assert payload.flat_tags == [
        "content:助眠",
        "content:放松",
        "content:白噪音",
    ]


def test_audio_documents_same_when_only_vector_id_differs() -> None:
    desired = build_compare_snapshot(
        audio_url="https://cdn.example.com/a.mp3",
        audio_name="海浪",
        flat_tags=["content:白噪音"],
        evidence_level="C",
        recommend_weight=0.45,
    )
    existing = {
        "audio_url": "https://cdn.example.com/a.mp3",
        "audio_name": "海浪",
        "evidence_level": "C",
        "recommend_weight": 0.45,
        "tags": {
            "sleep_stage": [],
            "content_form": [{"vector_id": "y", "label": "白噪音"}],
            "mechanism": [],
            "audio_feat": [],
            "rhythm": [],
            "risk_control": [],
        },
    }
    assert audio_documents_differ(desired, existing) is False
    assert desired == compare_snapshot_from_es_source(existing)


@pytest.mark.asyncio
async def test_sync_job_deletes_es_orphan(tmp_path) -> None:
    comm = MagicMock()
    comm.list_audio_materials_page = AsyncMock(return_value=([_make_material("id1")], 1))
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value={"id1", "orphan"})
    es_search.get_audio_source = AsyncMock(return_value=_existing_es_doc())
    es_sync = MagicMock()
    es_sync.delete_audio = AsyncMock()
    es_sync.upsert_audio = AsyncMock()

    settings = Settings(sync_backup_dir=str(tmp_path))
    result = await AudioEsSyncJob(comm, es_search, es_sync, settings).run()

    assert result.deleted == 1
    es_sync.delete_audio.assert_awaited_once_with("orphan")


@pytest.mark.asyncio
async def test_sync_job_skips_unchanged(tmp_path) -> None:
    comm = MagicMock()
    comm.list_audio_materials_page = AsyncMock(return_value=([_make_material("id1")], 1))
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value={"id1"})
    es_search.get_audio_source = AsyncMock(return_value=_existing_es_doc())
    es_sync = MagicMock()
    es_sync.upsert_audio = AsyncMock()

    settings = Settings(sync_backup_dir=str(tmp_path))
    result = await AudioEsSyncJob(comm, es_search, es_sync, settings).run()

    assert result.unchanged == 1
    es_sync.upsert_audio.assert_not_called()


@pytest.mark.asyncio
async def test_sync_job_replaces_previous_backup(tmp_path) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    backup_file = backup_dir / "audio_materials_backup.json"
    backup_file.write_text('{"old": true}', encoding="utf-8")

    comm = MagicMock()
    comm.list_audio_materials_page = AsyncMock(return_value=([_make_material("id1")], 1))
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value=set())
    es_search.get_audio_source = AsyncMock(return_value=None)
    es_sync = MagicMock()
    es_sync.upsert_audio = AsyncMock()

    await AudioEsSyncJob(comm, es_search, es_sync, Settings(sync_backup_dir=str(backup_dir))).run()

    content = backup_file.read_text(encoding="utf-8")
    assert "old" not in content
    assert "id1" in content


def test_build_payloads_skips_invalid_material() -> None:
    materials = [_make_material("ok"), _make_material("no-url", url="")]
    result = build_payloads_from_materials(materials)

    assert result.unique_count == 1
    assert result.skipped_invalid == 1
    assert result.deduped == 0
    assert "ok" in result.payloads


def test_build_payloads_dedupes_duplicate_id() -> None:
    materials = [_make_material("dup"), _make_material("dup")]
    result = build_payloads_from_materials(materials)

    assert result.unique_count == 1
    assert result.skipped_invalid == 0
    assert result.deduped == 1


@pytest.mark.asyncio
async def test_sync_job_reports_deduped_count(tmp_path) -> None:
    materials = [_make_material("dup"), _make_material("dup"), _make_material("unique")]
    comm = MagicMock()
    comm.list_audio_materials_page = AsyncMock(return_value=(materials, 3))
    es_search = MagicMock()
    es_search.list_all_audio_doc_ids = AsyncMock(return_value=set())
    es_search.get_audio_source = AsyncMock(return_value=None)
    es_sync = MagicMock()
    es_sync.upsert_audio = AsyncMock()

    result = await AudioEsSyncJob(
        comm, es_search, es_sync, Settings(sync_backup_dir=str(tmp_path))
    ).run()

    assert result.total_fetched == 3
    assert result.unique_sources == 2
    assert result.deduped == 1
    assert result.skipped_invalid == 0
    assert result.created == 2


def test_start_sync_scheduler_skipped_when_disabled() -> None:
    with patch("scripts.sync_es_from_comm.AsyncIOScheduler") as mock_cls:
        disabled = Settings(sync_enabled=False)
        start_sync_scheduler(AppState(settings=disabled), disabled)
        mock_cls.assert_not_called()
