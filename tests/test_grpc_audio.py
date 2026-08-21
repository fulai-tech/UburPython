"""功能手板 AudioRpc。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.schemas.audio import SearchAudioData
from app.server.handboard.audio.rpc import AudioRpc
from app.uburnode_grpc.grpc_gen import uburnode_pb2


def _context() -> MagicMock:
    ctx = MagicMock()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError)
    return ctx


@pytest.mark.asyncio
async def test_create_audio_maps_response() -> None:
    service = MagicMock()
    service.create_audio = AsyncMock(
        return_value={
            "id": "m1",
            "audio_name": "夜雨",
            "description": "",
            "status": True,
            "create_time": "",
            "update_time": "",
            "audio_url": "",
            "operation_type": 0,
            "created_by": "",
            "updated_by": "",
            "sleep_stage_tags": [],
            "content_form_tags": [],
            "mechanism_tags": [],
            "audio_engineering_tags": [],
            "medical_risk_tags": [],
            "evidence_level_tags": [],
            "embedding": [0.1, 0.2],
        }
    )
    rpc = AudioRpc(service)
    res = await rpc.CreateAudio(uburnode_pb2.CreateAudioReq(audio_name="夜雨"), _context())
    assert res.material.id == "m1"
    assert list(res.material.embedding) == [0.1, 0.2]


@pytest.mark.asyncio
async def test_update_delete_ok() -> None:
    service = MagicMock()
    service.update_audio = AsyncMock()
    service.delete_audio = AsyncMock()
    rpc = AudioRpc(service)
    ctx = _context()
    upd = await rpc.UpdateAudio(
        uburnode_pb2.UpdateAudioReq(material_id="m1", description="x"),
        ctx,
    )
    assert upd.ok is True
    deleted = await rpc.DeleteAudio(uburnode_pb2.IdRequest(id="m1"), ctx)
    assert deleted.ok is True


@pytest.mark.asyncio
async def test_search_returns_structs() -> None:
    service = MagicMock()
    service.search_audio = AsyncMock(
        return_value=SearchAudioData(materials=[{"id": "m1", "audio_name": "a"}])
    )
    rpc = AudioRpc(service)
    res = await rpc.SearchAudio(uburnode_pb2.SearchAudioReq(), _context())
    assert res.materials[0].fields["id"].string_value == "m1"
