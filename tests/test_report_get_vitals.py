"""GetVitals 三种查询模式。"""

from __future__ import annotations

from datetime import datetime, timezone
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
    assert payload["hr"]["value"] == 70.0
    assert len(payload["hr"]["series"]) == 2
    assert payload["hr"]["series"][0]["value"] == 72.0
    assert payload["hrv"]["value"] == 39.0
    assert payload["brv"]["personal_baseline"] == 50.0
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
    assert payload["hr"] == {"value": 0.0, "series": []}
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
    # 点值先 floor：72 / 69；均值 (72+69)/2=70.5 → 70
    assert payload["hr"]["series"][0]["value"] == 72.0
    assert payload["hr"]["series"][1]["value"] == 69.0
    assert payload["hr"]["value"] == 70.0
    assert payload["br"]["series"][0]["value"] == 16.0
    assert payload["br"]["value"] == 15.0


@pytest.mark.asyncio
async def test_get_vitals_invalid_mode() -> None:
    service = ReportService(None, Settings(), store=MagicMock())
    with pytest.raises(AppError):
        await service.get_vitals(uid="u1", device_id="d1")
