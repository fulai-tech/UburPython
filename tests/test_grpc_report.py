"""量产 ReportRpc。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from google.protobuf.json_format import MessageToDict

from app.server.somni.report.rpc import ReportRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _context() -> MagicMock:
    ctx = MagicMock()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError)
    return ctx


def _make_rpc() -> tuple[ReportRpc, MagicMock]:
    service = MagicMock()
    service.get_summary = AsyncMock(
        return_value={
            "sleep_summary": {
                "body_battery": 90,
                "body_battery_status": "Energy At Its Peak",
                "total_minutes": 457,
                "deep_sleep_minutes": 95,
                "avg_heart_rate": 62,
                "avg_respiratory_rate": 15,
            }
        }
    )
    service.get_events = AsyncMock(
        return_value={
            "record_date": "2026-08-24",
            "sleep_events": [],
            "event_count": 0,
            "abnormal_count": 0,
            "intervention_count": 0,
            "idf_data": [],
            "physio_data": [],
            "env_data": [],
        }
    )
    service.get_environment = AsyncMock(
        return_value={
            "environment_summary": {
                "temperature": {"value": 23, "min": 21, "max": 25},
                "humidity": {"value": 52, "min": 40, "max": 60},
                "illuminance": {"value": 1, "min": 0, "max": 5},
                "noise": {"value": 28, "min": 20, "max": 35},
            }
        }
    )
    service.get_structure = AsyncMock(
        return_value={
            "sleep_structure": {
                "awake": {"minutes": 20, "percent": 4},
                "rem_sleep": {"minutes": 100, "percent": 20},
                "light_sleep": {"minutes": 250, "percent": 50},
                "deep_sleep": {"minutes": 90, "percent": 18},
            }
        }
    )
    service.get_sleep_quality = AsyncMock(
        return_value={
            "sleep_quality": {
                "time_in_bed_minutes": 500,
                "sleep_onset_latency_minutes": 20,
                "sleep_efficiency": 92,
                "bedtime": "23:30",
                "wake_up_time": "07:30",
                "awake_after_onset_minutes": 18,
            }
        }
    )
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


@pytest.mark.asyncio
async def test_get_summary_maps_fields() -> None:
    rpc, _service = _make_rpc()
    res = await rpc.GetSummary(
        uburnode_somni_pb2.ReportDateReq(uid="u1", record_date="2026-08-24"),
        _context(),
    )
    payload = MessageToDict(res, preserving_proto_field_name=True)
    assert payload["sleep_summary"]["avg_heart_rate"] == 62
    assert payload["sleep_summary"]["body_battery_status"] == "Energy At Its Peak"
