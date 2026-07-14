"""EsSync HTTP CUD 写路径单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.sync import EsSync


@pytest.mark.asyncio
async def test_upsert_audio_skips_es_write() -> None:
    client = MagicMock()
    client.index = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    es_sync = EsSync(client, encoder, Settings())

    await es_sync.upsert_audio(
        "674a1b2c3d4e5f6789012345",
        audio_url="https://cdn.example.com/a.mp3",
        audio_name="测试",
        flat_tags=["content:雨声"],
        evidence_level="C",
        recommend_weight=0.45,
    )

    client.index.assert_not_called()


@pytest.mark.asyncio
async def test_delete_audio_removes_from_somni_index() -> None:
    client = MagicMock()
    client.delete = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    es_sync = EsSync(client, encoder, Settings())

    await es_sync.delete_audio("674a1b2c3d4e5f6789012345")

    client.delete.assert_awaited_once()
    assert client.delete.await_args.kwargs["index"] == Settings().es_audio_index
    assert client.delete.await_args.kwargs["id"] == "674a1b2c3d4e5f6789012345"


@pytest.mark.asyncio
async def test_upsert_audio_with_description_still_skips_es_write() -> None:
    client = MagicMock()
    client.index = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    encoder.encode_one = AsyncMock(return_value=[0.2] * 512)
    es_sync = EsSync(client, encoder, Settings())

    await es_sync.upsert_audio(
        "audio_001",
        audio_url="https://cdn.example.com/a.mp3",
        audio_name="深夜雨声",
        description="适合睡前放松",
        flat_tags=["content:雨声"],
        evidence_level="B",
        recommend_weight=0.75,
    )

    client.index.assert_not_called()
    encoder.encode_one.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_somni_material_indexes_when_audio_url_present() -> None:
    client = MagicMock()
    client.index = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    encoder.encode_one = AsyncMock(return_value=[0.2] * 512)
    es_sync = EsSync(client, encoder, Settings())

    await es_sync.upsert_somni_material(
        "audio_001",
        {
            "id": "audio_001",
            "audio_name": "深夜雨声",
            "audio_url": "https://cdn.example.com/a.mp3",
            "description": "适合睡前放松",
            "sleep_stage_tags": [{"name": "放松", "code": "unwind"}],
        },
    )

    client.index.assert_awaited_once()
    kwargs = client.index.await_args.kwargs
    assert kwargs["id"] == "audio_001"
    assert kwargs["document"]["audio_name"] == "深夜雨声"
    assert "id" not in kwargs["document"]
    assert "description_text" in kwargs["document"]
    assert kwargs["document"]["description_vector"] == [0.2] * 512
    encoder.encode_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_somni_material_skips_when_audio_url_missing() -> None:
    client = MagicMock()
    client.index = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    encoder.encode_one = AsyncMock(return_value=[0.2] * 512)
    es_sync = EsSync(client, encoder, Settings())

    await es_sync.upsert_somni_material(
        "audio_001",
        {"audio_name": "仅名称", "audio_url": ""},
    )

    client.index.assert_not_called()
    encoder.encode_one.assert_not_called()
