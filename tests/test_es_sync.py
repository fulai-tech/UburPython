"""EsSync 标签向量去重复用逻辑单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.sync import EsSync


@pytest.mark.asyncio
async def test_embed_and_store_tag_reuses_existing_vector_id() -> None:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={"hits": {"hits": [{"_id": "existing_vector_id_24ch"}]}}
    )
    client.index = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    encoder.encode_one = AsyncMock()

    es_sync = EsSync(client, encoder, Settings())
    vector_id = await es_sync._embed_and_store_tag("content_form", "冥想")

    assert vector_id == "existing_vector_id_24ch"
    encoder.encode_one.assert_not_called()
    client.index.assert_not_called()


@pytest.mark.asyncio
async def test_embed_and_store_tag_creates_when_label_not_found() -> None:
    client = MagicMock()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    client.index = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    encoder.encode_one = AsyncMock(return_value=[0.1] * 512)

    es_sync = EsSync(client, encoder, Settings())
    vector_id = await es_sync._embed_and_store_tag("content_form", "新标签")

    assert len(vector_id) == 24
    encoder.encode_one.assert_awaited_once_with("新标签")
    client.index.assert_awaited_once()
    indexed = client.index.await_args.kwargs
    assert indexed["document"]["label_text"] == "新标签"
    assert indexed["document"]["dimension"] == "content_form"


@pytest.mark.asyncio
async def test_delete_audio_only_removes_audio_index() -> None:
    client = MagicMock()
    client.delete = AsyncMock()
    encoder = MagicMock(spec=Encoder)
    es_sync = EsSync(client, encoder, Settings())

    await es_sync.delete_audio("674a1b2c3d4e5f6789012345")

    client.delete.assert_awaited_once()
    assert client.delete.await_args.kwargs["index"] == Settings().es_audio_index
    assert client.delete.await_args.kwargs["id"] == "674a1b2c3d4e5f6789012345"
