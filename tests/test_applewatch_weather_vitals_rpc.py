"""AppleWatch / Weather / GetVitals RPC 适配。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from google.protobuf.json_format import MessageToDict

from app.server.somni.applewatch.rpc import AppleWatchRpc
from app.server.somni.report.rpc import ReportRpc
from app.server.somni.weather.rpc import WeatherRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _context() -> MagicMock:
    ctx = MagicMock()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError)
    return ctx


@pytest.mark.asyncio
async def test_weather_rpc_fixed() -> None:
    service = MagicMock()
    service.get_weather = AsyncMock(
        return_value={
            "weather": {
                "date": "2026-09-24",
                "temperature": {"min": 15, "max": 29},
                "condition_code": "partly_cloudy_rain",
                "sunrise_at": "2026-09-24T07:00:00+08:00",
                "sunset_at": "2026-09-24T20:00:00+08:00",
                "uv_index": {"level": "low"},
                "humidity_percent": 58,
                "aqi": "hazardous",
            }
        }
    )
    rpc = WeatherRpc(service)
    res = await rpc.GetWeather(
        uburnode_somni_pb2.GetWeatherReq(
            record_date="2026-09-24",
            uid="u1",
            timezone="Asia/Shanghai",
        ),
        _context(),
    )
    data = MessageToDict(res, preserving_proto_field_name=True)
    assert data["weather"]["aqi"] == "hazardous"


@pytest.mark.asyncio
async def test_applewatch_rpc() -> None:
    service = MagicMock()
    service.get_daily_aggregate = AsyncMock(
        return_value={
            "weather": {
                "date": "2026-09-24",
                "temperature": {"min": 15, "max": 29},
                "condition_code": "partly_cloudy_rain",
                "sunrise_at": "a",
                "sunset_at": "b",
                "uv_index": {"level": "low"},
                "humidity_percent": 58,
                "aqi": "hazardous",
            },
            "steps": {
                "date": "2026-09-24",
                "total": 100,
                "series": [{"start_at": "t", "value": 100}],
                "tonight_risk": {
                    "level": "low",
                    "display_text": "LOW RISKS FOR TONIGHT",
                },
            },
            "emotion_records": {"state": "Slightly Stressful", "stage": "Extended Unwind"},
            "schedule": {
                "schedule_name": "Morning meeting",
                "start_at": "9:00",
                "current_clock": "7:20",
            },
        }
    )
    rpc = AppleWatchRpc(service)
    res = await rpc.GetDailyAggregate(
        uburnode_somni_pb2.GetDailyAggregateReq(
            start_time="2026-09-24T00:00:00+08:00",
            end_time="2026-09-25T00:00:00+08:00",
            uid="u1",
            device_id="d1",
            timezone="Asia/Shanghai",
        ),
        _context(),
    )
    data = MessageToDict(res, preserving_proto_field_name=True)
    assert data["steps"]["tonight_risk"]["display_text"] == "LOW RISKS FOR TONIGHT"
    assert data["schedule"]["current_clock"] == "7:20"


@pytest.mark.asyncio
async def test_get_vitals_rpc() -> None:
    service = MagicMock()
    service.get_vitals = AsyncMock(
        return_value={
            "hr": {"value": 68.0, "series": [{"collected_at": "t", "value": 68.0}]},
            "br": {"value": 15.0, "series": []},
            "hrv": {
                "value": 39.0,
                "personal_baseline": 35.0,
                "vs_baseline_percent": 12.0,
                "series": [],
            },
            "brv": {
                "value": 56.0,
                "personal_baseline": 50.0,
                "vs_baseline_percent": 12.0,
                "series": [],
            },
        }
    )
    rpc = ReportRpc(service)
    res = await rpc.GetVitals(
        uburnode_somni_pb2.GetVitalsReq(session_id="s1"),
        _context(),
    )
    data = MessageToDict(res, preserving_proto_field_name=True)
    assert data["hr"]["value"] == 68.0
    assert data["hrv"]["value"] == 39.0
