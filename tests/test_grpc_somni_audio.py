"""量产 AudioRpc。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
async def test_get_audio_passes_page_and_query() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "materials": [{"id": "m1", "audio_name": "雨声"}],
            "page": {"page": 1, "page_size": 20, "total": 1, "total_pages": 1},
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
    )
    assert res.materials[0]["audio_name"] == "雨声"
    assert res.page.total == 1


@pytest.mark.asyncio
async def test_get_audio_fetch_all() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "materials": [],
            "page": {"page": 1, "page_size": 0, "total": 0, "total_pages": 1},
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
    )


@pytest.mark.asyncio
async def test_get_audio_passes_tag_code() -> None:
    rpc, service = _make_rpc()
    service.get_audio = AsyncMock(
        return_value={
            "materials": [],
            "page": {"page": 1, "page_size": 20, "total": 0, "total_pages": 0},
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
                }
            ]
        }
    )
    res = await rpc.GetAudioTag(uburnode_somni_pb2.GetAudioTagReq(), _context())
    service.get_audio_tag.assert_awaited_once_with()
    assert res.tags[0].code == "natural_sound"
    assert res.tags[0].name_en == "Natural Sound"


@pytest.mark.asyncio
async def test_get_hot_no_args() -> None:
    rpc, service = _make_rpc()
    res = await rpc.GetHot(uburnode_somni_pb2.GetHotReq(), _context())
    service.get_hot.assert_awaited_once_with()
    assert res == uburnode_somni_pb2.GetHotRes()
