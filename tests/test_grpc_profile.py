"""量产 ProfileRpc。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.core.codes import HttpStatus
from app.core.exceptions import AppError
from app.server.somni.profile.rpc import ProfileRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _context() -> MagicMock:
    ctx = MagicMock()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError)
    return ctx


def _make_rpc() -> tuple[ProfileRpc, MagicMock]:
    service = MagicMock()
    service.get_user_profile = AsyncMock(
        return_value={"profile": {"sleep_type": "sleep_onset_insomnia"}}
    )
    return ProfileRpc(service), service


@pytest.mark.asyncio
async def test_get_user_profile_requires_uid() -> None:
    rpc, _service = _make_rpc()
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetUserProfile(
            uburnode_somni_pb2.GetUserProfileReq(uid="", type="long_terms"),
            _context(),
        )


@pytest.mark.asyncio
async def test_get_user_profile_requires_type() -> None:
    rpc, _service = _make_rpc()
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetUserProfile(
            uburnode_somni_pb2.GetUserProfileReq(uid="u1", type=""),
            _context(),
        )


@pytest.mark.asyncio
async def test_short_terms_requires_record_date() -> None:
    rpc, _service = _make_rpc()
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetUserProfile(
            uburnode_somni_pb2.GetUserProfileReq(uid="u1", type="short_terms"),
            _context(),
        )


@pytest.mark.asyncio
async def test_get_user_profile_maps_json() -> None:
    rpc, service = _make_rpc()
    res = await rpc.GetUserProfile(
        uburnode_somni_pb2.GetUserProfileReq(uid="u1", type="long_terms"),
        _context(),
    )
    service.get_user_profile.assert_awaited_once_with("u1", "long_terms", "")
    payload = json.loads(res.profile)
    assert payload["sleep_type"] == "sleep_onset_insomnia"


@pytest.mark.asyncio
async def test_get_user_profile_short_terms() -> None:
    rpc, service = _make_rpc()
    await rpc.GetUserProfile(
        uburnode_somni_pb2.GetUserProfileReq(
            uid="u1", type="short_terms", record_date="2026-09-01"
        ),
        _context(),
    )
    service.get_user_profile.assert_awaited_once_with(
        "u1", "short_terms", "2026-09-01"
    )


@pytest.mark.asyncio
async def test_get_user_profile_not_found_aborts() -> None:
    service = MagicMock()
    service.get_user_profile = AsyncMock(
        side_effect=AppError(message="画像不存在：u1", status_code=HttpStatus.NOT_FOUND)
    )
    rpc = ProfileRpc(service)
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetUserProfile(
            uburnode_somni_pb2.GetUserProfileReq(uid="u1", type="long_terms"),
            _context(),
        )
