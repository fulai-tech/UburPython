"""量产 AudioRpc。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.server.somni.audio.rpc import AudioRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _context() -> MagicMock:
    return MagicMock()


def _make_rpc() -> tuple[AudioRpc, MagicMock]:
    service = MagicMock()
    service.get_audio = AsyncMock()
    service.get_audio_tag = AsyncMock()
    service.get_hot = AsyncMock()
    return AudioRpc(service), service


@pytest.mark.asyncio
async def test_get_audio_passes_language_en() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={"list": [], "page": 1, "page_size": 20, "total": 0}
    )
    req = uburnode_somni_pb2.GetAudioReq(page=1, page_size=20, language="en")
    await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=1,
        page_size=20,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="en",
    )


@pytest.mark.asyncio
async def test_get_audio_rejects_invalid_language() -> None:
    rpc, service = _make_rpc()
    ctx = _context()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError())
    req = uburnode_somni_pb2.GetAudioReq(language="fr")
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetAudio(req, ctx)
    service.get_audio.assert_not_called()
    ctx.abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_audio_passes_page_and_query() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "list": [
                {
                    "id": "m1",
                    "audio_name": "雨声",
                    "audio_url": "https://cdn.example/a.mp3",
                    "cover_url": "https://cdn.example/a.png",
                    "description": "desc",
                    "vip": 0,
                    "tag": ["自然声", "中雨/稳定雨声"],
                }
            ],
            "page": 1,
            "page_size": 20,
            "total": 1,
        }
    )
    req = uburnode_somni_pb2.GetAudioReq(page=1, page_size=20, query_text="雨声")
    res = await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=1,
        page_size=20,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
        language="zh",
    )
    assert res.list[0].audio_name == "雨声"
    assert res.list[0].vip == 0
    assert res.total == 1
    assert res.page == 1
    assert res.page_size == 20
    assert list(res.list[0].tag) == ["自然声", "中雨/稳定雨声"]


@pytest.mark.asyncio
async def test_get_audio_fetch_all() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "list": [],
            "page": 1,
            "page_size": 0,
            "total": 0,
        }
    )
    req = uburnode_somni_pb2.GetAudioReq(fetch_all=True)
    await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=None,
        page_size=None,
        fetch_all=True,
        query_text="",
        tag_code="",
        language="zh",
    )


@pytest.mark.asyncio
async def test_get_audio_passes_tag_code() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "list": [],
            "page": 1,
            "page_size": 20,
            "total": 0,
        }
    )
    req = uburnode_somni_pb2.GetAudioReq(tag_code="steady_rain")
    await rpc.GetAudio(req, _context())
    service.get_audio.assert_awaited_once_with(
        page=None,
        page_size=None,
        fetch_all=False,
        query_text="",
        tag_code="steady_rain",
        language="zh",
    )


@pytest.mark.asyncio
async def test_get_audio_tag_maps_fields() -> None:
    rpc, service = _make_rpc()
    service.get_audio_tag = AsyncMock(
        return_value={
            "tags": [
                {
                    "type": "content_form",
                    "code": "natural_sound",
                    "name": "自然声",
                    "name_en": "Natural Sound",
                    "id": "root-natural",
                    "parent_tag_id": None,
                    "status": "启用",
                }
            ]
        }
    )
    res = await rpc.GetAudioTag(uburnode_somni_pb2.GetAudioTagReq(), _context())
    service.get_audio_tag.assert_awaited_once_with()
    assert res.tags[0].code == "natural_sound"
    assert res.tags[0].name_en == "Natural Sound"
    assert res.tags[0].id == "root-natural"
    assert res.tags[0].parent_tag_id.WhichOneof("kind") == "null_value"
    assert res.tags[0].status == "启用"


@pytest.mark.asyncio
async def test_get_hot_passes_language_en() -> None:
    rpc, service = _make_rpc()
    service.get_hot = AsyncMock(return_value={"items": []})
    await rpc.GetHot(uburnode_somni_pb2.GetHotReq(language="en"), _context())
    service.get_hot.assert_awaited_once_with(language="en", kind="query")


@pytest.mark.asyncio
async def test_get_hot_passes_kind_tag() -> None:
    rpc, service = _make_rpc()
    service.get_hot = AsyncMock(return_value={"items": []})
    await rpc.GetHot(
        uburnode_somni_pb2.GetHotReq(language="zh", kind="tag"),
        _context(),
    )
    service.get_hot.assert_awaited_once_with(language="zh", kind="tag")


@pytest.mark.asyncio
async def test_get_hot_maps_items() -> None:
    rpc, service = _make_rpc()
    service.get_hot = AsyncMock(
        return_value={"items": [{"keyword": "雨声", "score": 5}]}
    )
    res = await rpc.GetHot(uburnode_somni_pb2.GetHotReq(), _context())
    service.get_hot.assert_awaited_once_with(language="zh", kind="query")
    assert res.items[0].keyword == "雨声"
    assert res.items[0].score == 5


@pytest.mark.asyncio
async def test_get_hot_rejects_invalid_kind() -> None:
    rpc, service = _make_rpc()
    ctx = _context()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError())
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetHot(uburnode_somni_pb2.GetHotReq(kind="other"), ctx)
    service.get_hot.assert_not_called()
    ctx.abort.assert_awaited_once()
