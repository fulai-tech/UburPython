"""GetVitals 三种查询模式。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AppError
from app.server.somni.report.service import ReportService


@pytest.mark.asyncio
async def test_get_vitals_by_session() -> None:
    store = MagicMock()
    store.list_telemetry_by_filter = AsyncMock(
        return_value=[
            {
                "ts": datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc),
                "data": {"hr": 72, "br": 16},
            },
            {
                "ts": datetime(2026, 9, 23, 17, 0, tzinfo=timezone.utc),
                "data": {"hr": 68, "br": 14},
            },
        ]
    )
    service = ReportService(None, Settings(), store=store)
    payload = await service.get_vitals(session_id="session-001")
    assert payload["hr"]["value"] == 70
    assert isinstance(payload["hr"]["value"], int)
    assert len(payload["hr"]["series"]) == 2
    assert payload["hr"]["series"][0]["value"] == 72
    assert isinstance(payload["hr"]["series"][0]["value"], int)
    assert payload["hrv"]["value"] == 39
    assert isinstance(payload["hrv"]["value"], int)
    assert payload["brv"]["personal_baseline"] == 50
    assert isinstance(payload["brv"]["series"][0]["value"], int)
    query = store.list_telemetry_by_filter.await_args.args[0]
    assert query == {"session_id": "session-001", "metric": "sleep"}


@pytest.mark.asyncio
async def test_get_vitals_by_uid_window() -> None:
    store = MagicMock()
    store.list_device_ids_by_uid = AsyncMock(return_value=["d1", "d2"])
    store.list_telemetry_by_filter = AsyncMock(return_value=[])
    service = ReportService(None, Settings(), store=store)
    payload = await service.get_vitals(
        uid="user-001",
        start_time="2026-09-23T23:00:00+08:00",
        end_time="2026-09-24T07:00:00+08:00",
    )
    assert payload["hr"] == {"value": 0, "series": []}
    assert isinstance(payload["hr"]["value"], int)
    query = store.list_telemetry_by_filter.await_args.args[0]
    assert query["device_id"] == {"$in": ["d1", "d2"]}
    assert query["metric"] == "sleep"


@pytest.mark.asyncio
async def test_get_vitals_by_device_window() -> None:
    store = MagicMock()
    store.list_telemetry_by_filter = AsyncMock(return_value=[])
    service = ReportService(None, Settings(), store=store)
    await service.get_vitals(
        device_id="device-001",
        start_time="2026-09-23T23:00:00+08:00",
        end_time="2026-09-24T07:00:00+08:00",
    )
    query = store.list_telemetry_by_filter.await_args.args[0]
    assert query["device_id"] == "device-001"


@pytest.mark.asyncio
async def test_get_vitals_floors_average() -> None:
    store = MagicMock()
    store.list_telemetry_by_filter = AsyncMock(
        return_value=[
            {
                "ts": datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc),
                "data": {"hr": 72.9, "br": 16.8},
            },
            {
                "ts": datetime(2026, 9, 23, 17, 0, tzinfo=timezone.utc),
                "data": {"hr": 69.1, "br": 14.2},
            },
        ]
    )
    service = ReportService(None, Settings(), store=store)
    payload = await service.get_vitals(session_id="session-floor")
    # 单点一桶：72.9→73，69.1→69；整段 value 仍是各点先 floor 再 floor 均值
    assert payload["hr"]["series"][0]["value"] == 73
    assert payload["hr"]["series"][1]["value"] == 69
    assert payload["hr"]["value"] == 70
    assert payload["br"]["series"][0]["value"] == 17
    assert payload["br"]["value"] == 15
    assert isinstance(payload["hr"]["series"][0]["value"], int)
    assert payload["hrv"]["series"][0]["value"] == 36


@pytest.mark.asyncio
async def test_get_vitals_series_half_hour_buckets() -> None:
    store = MagicMock()
    anchor = datetime(2026, 9, 23, 15, 7, tzinfo=timezone.utc)
    store.list_telemetry_by_filter = AsyncMock(
        return_value=[
            {"ts": anchor, "data": {"hr": 60.4, "br": 14.2}},
            {
                "ts": anchor + timedelta(minutes=20),
                "data": {"hr": 70.6, "br": 15.6},
            },
            {
                "ts": anchor + timedelta(minutes=30),
                "data": {"hr": 2.5, "br": 16.0},
            },
            {
                "ts": anchor + timedelta(minutes=90),
                "data": {"hr": 80.0, "br": 18.0},
            },
        ]
    )
    service = ReportService(None, Settings(), store=store)
    payload = await service.get_vitals(session_id="session-bucket")
    assert [point["collected_at"] for point in payload["hr"]["series"]] == [
        "2026-09-23T23:07:00+08:00",
        "2026-09-23T23:37:00+08:00",
        "2026-09-24T00:37:00+08:00",
    ]
    # 60.4 与 70.6 同桶，均值 65.5 → 66；2.5 单独成桶 → 3（不是银行家舍入的 2）
    assert [point["value"] for point in payload["hr"]["series"]] == [66, 3, 80]
    assert [point["value"] for point in payload["br"]["series"]] == [15, 16, 18]
    assert all(isinstance(point["value"], int) for point in payload["hr"]["series"])
    assert payload["hr"]["value"] == 53
    assert isinstance(payload["hr"]["value"], int)


@pytest.mark.asyncio
async def test_get_vitals_invalid_mode() -> None:
    service = ReportService(None, Settings(), store=MagicMock())
    with pytest.raises(AppError):
        await service.get_vitals(uid="u1", device_id="d1")
