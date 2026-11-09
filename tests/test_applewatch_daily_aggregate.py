"""Apple Watch 聚合与 tonight_risk。"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from app.core.config import Settings
from app.server.somni.applewatch.service import (
    AppleWatchService,
    tonight_risk_for_total,
    week_day_from_instant,
    week_day_from_record_date,
)


def test_week_day_monday_is_1() -> None:
    assert week_day_from_record_date("2026-09-28") == 1  # 周一
    assert (
        week_day_from_instant(
            datetime(2026, 9, 27, 16, 0, tzinfo=ZoneInfo("UTC")),
            "Asia/Shanghai",
        )
        == 1
    )


def test_tonight_risk_levels() -> None:
    assert tonight_risk_for_total(0)["level"] == "low"
    assert tonight_risk_for_total(0)["display_text"] == "LOW RISKS FOR TONIGHT"
    assert tonight_risk_for_total(6420)["level"] == "medium"
    assert tonight_risk_for_total(12000)["level"] == "excellent"


@pytest.mark.asyncio
async def test_get_daily_aggregate_builds_payload() -> None:
    store = MagicMock()
    store.list_step_points = AsyncMock(
        return_value=[
            {
                "sample_key": "a",
                "original_type": "STEPS",
                "value_numeric": 2100,
                "start_at": datetime(2026, 9, 24, 4, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                "end_at": datetime(2026, 9, 24, 4, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
            },
            {
                "sample_key": "b",
                "original_type": "STEPS",
                "value_numeric": 900,
                "start_at": datetime(2026, 9, 24, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                "end_at": datetime(2026, 9, 24, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
            },
        ]
    )
    store.find_alarm = AsyncMock(return_value={"wake_time": "7:20"})

    service = AppleWatchService(None, Settings(), store=store)
    payload = await service.get_daily_aggregate(
        start_time="2026-09-24T00:00:00+08:00",
        end_time="2026-09-25T00:00:00+08:00",
        uid="user-001",
        device_id="device-001",
        timezone="Asia/Shanghai",
    )

    store.list_step_points.assert_awaited_once()
    store.find_alarm.assert_awaited_once()
    assert payload["weather"]["condition_code"] == "partly_cloudy_rain"
    assert payload["steps"]["total"] == 3000
    assert payload["steps"]["date"] == "2026-09-24"
    assert payload["steps"]["tonight_risk"]["level"] == "low"
    assert payload["steps"]["tonight_risk"]["display_text"] == "LOW RISKS FOR TONIGHT"
    assert payload["emotion_records"]["state"] == "Slightly Stressful"
    # 日程固定，不读日历
    assert payload["schedule"]["schedule_name"] == "Morning meeting"
    assert payload["schedule"]["current_clock"] == "7:20"
    assert payload["schedule"]["start_at"] == "9:00"
