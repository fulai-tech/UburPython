"""AudioService：直连 Mongo + ES 编排测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.audio import (
    CreateAudioRequest,
    SearchAudioRequest,
    UpdateAudioRequest,
)
from app.server.handboard.audio.service import AudioService


def _service(
    *,
    materials: MagicMock | None = None,
    es_sync: MagicMock | None = None,
    retrieval: MagicMock | None = None,
    search_cache: MagicMock | None = None,
) -> AudioService:
    retrieval_svc = retrieval or MagicMock()
    if retrieval is None:
        retrieval_svc.clear_sleep_stage_cache = AsyncMock()
        retrieval_svc.warm_sleep_stage_cache = AsyncMock()
    return AudioService(
        materials,
        es_sync or MagicMock(),
        retrieval_svc,
        search_cache=search_cache,
        sleep_stage_rewarm_delay_sec=0,
    )


@pytest.mark.asyncio
async def test_create_audio_writes_mongo_then_es() -> None:
    materials = MagicMock()
    materials.insert_material = AsyncMock(
        return_value={
            "id": "abc123",
            "audio_name": "雨声",
            "audio_url": "https://cdn.example.com/a.mp3",
        }
    )
    es_sync = MagicMock()
    es_sync.upsert_somni_material = AsyncMock()
    search_cache = MagicMock()
    search_cache.clear_all = AsyncMock()
    service = _service(materials=materials, es_sync=es_sync, search_cache=search_cache)

    result = await service.create_audio(
        CreateAudioRequest.model_validate(
            {"audio_name": "雨声", "audio_url": "https://cdn.example.com/a.mp3"}
        )
    )

    assert result["id"] == "abc123"
    materials.insert_material.assert_awaited_once()
    es_sync.upsert_somni_material.assert_awaited_once()
    assert es_sync.upsert_somni_material.await_args.args[0] == "abc123"
    search_cache.clear_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_audio_writes_mongo_then_es() -> None:
    materials = MagicMock()
    materials.update_material = AsyncMock(
        return_value={"id": "m1", "description": "新描述"}
    )
    es_sync = MagicMock()
    es_sync.upsert_somni_material = AsyncMock()
    search_cache = MagicMock()
    search_cache.clear_all = AsyncMock()
    service = _service(materials=materials, es_sync=es_sync, search_cache=search_cache)

    await service.update_audio(
        "m1", UpdateAudioRequest.model_validate({"description": "新描述"})
    )
    materials.update_material.assert_awaited_once()
    es_sync.upsert_somni_material.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_audio_deletes_mongo_and_es() -> None:
    materials = MagicMock()
    materials.delete_material = AsyncMock()
    es_sync = MagicMock()
    es_sync.delete_audio = AsyncMock()
    search_cache = MagicMock()
    search_cache.clear_all = AsyncMock()
    service = _service(materials=materials, es_sync=es_sync, search_cache=search_cache)

    await service.delete_audio("m1")
    materials.delete_material.assert_awaited_once_with("m1")
    es_sync.delete_audio.assert_awaited_once_with("m1")


@pytest.mark.asyncio
async def test_search_audio_uses_retrieval() -> None:
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[{"id": "m1"}])
    retrieval.clear_sleep_stage_cache = AsyncMock()
    retrieval.warm_sleep_stage_cache = AsyncMock()
    service = _service(retrieval=retrieval)
    data = await service.search_audio(SearchAudioRequest(query_text="雨"))
    assert data.materials == [{"id": "m1"}]
