"""AudioService 创建/更新写 Mongo + ES 编排测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.audio import CreateAudioRequest, UpdateAudioRequest
from app.services.audio import AudioService


def _service(
    *,
    materials: MagicMock | None = None,
    es_sync: MagicMock | None = None,
    retrieval: MagicMock | None = None,
    comm: MagicMock | None = None,
) -> AudioService:
    return AudioService(
        comm or MagicMock(),
        es_sync or MagicMock(),
        retrieval or MagicMock(),
        materials=materials,
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
    service = _service(materials=materials, es_sync=es_sync)

    result = await service.create_audio(
        CreateAudioRequest.model_validate(
            {"audio_name": "雨声", "audio_url": "https://cdn.example.com/a.mp3"}
        )
    )

    assert result["id"] == "abc123"
    materials.insert_material.assert_awaited_once()
    es_sync.upsert_somni_material.assert_awaited_once_with(
        "abc123",
        {
            "id": "abc123",
            "audio_name": "雨声",
            "audio_url": "https://cdn.example.com/a.mp3",
        },
    )


@pytest.mark.asyncio
async def test_update_audio_partial_fields() -> None:
    materials = MagicMock()
    materials.update_material = AsyncMock(
        return_value={
            "id": "abc123",
            "audio_name": "雨声",
            "description": "新描述",
            "audio_url": "https://cdn.example.com/a.mp3",
        }
    )
    es_sync = MagicMock()
    es_sync.upsert_somni_material = AsyncMock()
    service = _service(materials=materials, es_sync=es_sync)

    await service.update_audio(
        "abc123",
        UpdateAudioRequest.model_validate({"description": "新描述"}),
    )

    materials.update_material.assert_awaited_once_with("abc123", {"description": "新描述"})
    es_sync.upsert_somni_material.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_audio_calls_comm_and_es() -> None:
    comm = MagicMock()
    comm.delete_audio_material = AsyncMock()
    es_sync = MagicMock()
    es_sync.delete_audio = AsyncMock()
    service = _service(comm=comm, es_sync=es_sync)

    await service.delete_audio("abc123")

    comm.delete_audio_material.assert_awaited_once_with("abc123")
    es_sync.delete_audio.assert_awaited_once_with("abc123")


@pytest.mark.asyncio
async def test_create_audio_without_mongo_raises() -> None:
    from app.core.exceptions import MongoNotConfiguredError

    service = _service(materials=None)
    with pytest.raises(MongoNotConfiguredError):
        await service.create_audio(CreateAudioRequest.model_validate({"audio_name": "x"}))
