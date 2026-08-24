"""量产 ReportRpc。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.server.somni.report.rpc import ReportRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _context() -> MagicMock:
    ctx = MagicMock()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError)
    return ctx


def _make_rpc() -> tuple[ReportRpc, MagicMock]:
    service = MagicMock()
    service.get_summary = AsyncMock()
    service.get_events = AsyncMock()
    service.get_environment = AsyncMock()
    service.get_structure = AsyncMock()
    service.get_sleep_quality = AsyncMock()
    return ReportRpc(service), service


@pytest.mark.asyncio
async def test_report_requires_uid_and_record_date() -> None:
    rpc, _service = _make_rpc()
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetSummary(
            uburnode_somni_pb2.ReportDateReq(uid="", record_date="2026-08-24"),
            _context(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "service_name"),
    [
        ("GetSummary", "get_summary"),
        ("GetEvents", "get_events"),
        ("GetEnvironment", "get_environment"),
        ("GetStructure", "get_structure"),
        ("GetSleepQuality", "get_sleep_quality"),
    ],
)
async def test_report_rpcs_call_service(method_name: str, service_name: str) -> None:
    rpc, service = _make_rpc()
    req = uburnode_somni_pb2.ReportDateReq(uid="u1", record_date="2026-08-24")
    res = await getattr(rpc, method_name)(req, _context())
    getattr(service, service_name).assert_awaited_once_with("u1", "2026-08-24")
    assert res is not None
