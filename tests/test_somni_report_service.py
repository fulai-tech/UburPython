"""量产 ReportService 业务单测。"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import MongoNotConfiguredError
from app.server.somni.report.service import ReportService


def _settings() -> Settings:
    return Settings(somni_mongo_uri="mongodb://localhost")


def _service(store: MagicMock) -> ReportService:
    return ReportService(client=MagicMock(), settings=_settings(), store=store)


@pytest.mark.asyncio
async def test_get_summary_computes_stages_and_avg() -> None:
    store = MagicMock()
    store.find_record = AsyncMock(
        return_value={
            "raw_data": {
                "bed_time": datetime(2026, 8, 9, 15, 30, tzinfo=UTC),
                "wake_up_time": datetime(2026, 8, 10, 0, 30, tzinfo=UTC),
                "deep_sleep_ratio": 20,
                "light_sleep_ratio": 50,
                "rem_ratio": 20,
                "awake_ratio": 10,
            }
        }
    )
    store.find_sleep_report = AsyncMock(
        return_value={
            "sleep_summary": {
                "body_battery": 90,
                "body_battery_status": "Energy At Its Peak",
            }
        }
    )
    store.find_device_id = AsyncMock(return_value="dev_1")
    store.list_telemetry = AsyncMock(
        return_value=[
            {"data": {"hr": 66.5, "br": 16}},
            {"data": {"hr": 62.2, "br": 14.8}},
        ]
    )
    svc = _service(store)
    payload = await svc.get_summary("u1", "2026-08-10")
    summary = payload["sleep_summary"]
    assert summary["body_battery"] == 90
    assert summary["body_battery_status"] == "Energy At Its Peak"
    assert summary["deep_sleep_minutes"] == 108
    assert summary["total_minutes"] == 108 + 270 + 108
    assert summary["avg_heart_rate"] == 64
    assert summary["avg_respiratory_rate"] == 15


@pytest.mark.asyncio
async def test_get_summary_without_device_zeros_avg() -> None:
    store = MagicMock()
    store.find_record = AsyncMock(return_value=None)
    store.find_sleep_report = AsyncMock(return_value=None)
    store.find_device_id = AsyncMock(return_value=None)
    svc = _service(store)
    summary = (await svc.get_summary("u1", "2026-08-10"))["sleep_summary"]
    assert summary == {
        "body_battery": 0,
        "body_battery_status": "",
        "total_minutes": 0,
        "deep_sleep_minutes": 0,
        "avg_heart_rate": 0,
        "avg_respiratory_rate": 0,
    }


@pytest.mark.asyncio
async def test_get_environment_stats() -> None:
    store = MagicMock()
    store.find_device_id = AsyncMock(return_value="dev_1")
    store.list_telemetry = AsyncMock(
        return_value=[
            {"data": {"temp": 21.9, "humi": 40.2, "lux": 1.2, "noise_db": 28.9}},
            {"data": {"temp": 25.8, "humi": 55.1, "lux": 4.9, "noise_db": 34.1}},
        ]
    )
    env = (await _service(store).get_environment("u1", "2026-08-10"))[
        "environment_summary"
    ]
    assert env["temperature"] == {"value": 23, "min": 21, "max": 25}
    assert env["noise"]["value"] == 31
    assert "status" not in env["temperature"]


@pytest.mark.asyncio
async def test_get_events_attaches_intervention() -> None:
    store = MagicMock()
    store.list_events = AsyncMock(
        return_value=[
            {
                "_id": "ab1",
                "event_time": "02:15",
                "type": "abnormal",
                "code": "heart_rate_increase",
                "events": [
                    {
                        "event_type": "心率上升",
                        "duration": "00:03:20",
                        "trigger_cause": "可能受到噪音影响",
                        "action_taken": "播放舒缓声音",
                        "result_summary": "心率恢复",
                    }
                ],
            },
            {
                "_id": "iv1",
                "type": "intervention",
                "related_event_id": "ab1",
                "event_time": "02:16",
                "event_type": "干预",
                "duration": "00:01:00",
                "trigger_cause": "噪音",
                "action_taken": "播放白噪音",
                "result_summary": "恢复",
            },
        ]
    )
    store.find_record = AsyncMock(
        return_value={"idf_data": [{"stage": "deep", "start": "01:00", "end": "02:20"}]}
    )
    store.find_device_id = AsyncMock(return_value="dev_1")
    store.list_telemetry = AsyncMock(
        side_effect=[
            [
                {
                    "ts": datetime(2026, 8, 9, 18, 15, tzinfo=UTC),
                    "data": {"hr": 68.2, "br": 16.1},
                }
            ],
            [
                {
                    "ts": datetime(2026, 8, 9, 18, 15, tzinfo=UTC),
                    "data": {"temp": 23.2, "humi": 52.9, "lux": 1.2, "noise_db": 28.4},
                }
            ],
        ]
    )
    payload = await _service(store).get_events("u1", "2026-08-10")
    assert payload["event_count"] == 1
    assert payload["abnormal_count"] == 1
    assert payload["intervention_count"] == 1
    assert len(payload["sleep_events"]) == 1
    assert payload["sleep_events"][0]["intervention"]["event_type"] == "干预"
    assert payload["idf_data"][0]["stage"] == "deep"
    assert payload["physio_data"][0]["metrics"]["heart_rate"] == 68
    assert payload["env_data"][0]["noise"] == 28


@pytest.mark.asyncio
async def test_get_structure_and_sleep_quality() -> None:
    store = MagicMock()
    store.find_record = AsyncMock(
        return_value={
            "raw_data": {
                "bed_time": datetime(2026, 8, 9, 15, 30, tzinfo=UTC),
                "sleep_time": datetime(2026, 8, 9, 15, 50, tzinfo=UTC),
                "wake_time": datetime(2026, 8, 10, 0, 12, tzinfo=UTC),
                "wake_up_time": datetime(2026, 8, 10, 0, 30, tzinfo=UTC),
                "awake_ratio": 10,
                "deep_sleep_ratio": 20,
                "light_sleep_ratio": 50,
                "rem_ratio": 20,
                "sleep_latency": 20,
                "sleep_efficiency": 92,
            }
        }
    )
    svc = _service(store)
    structure = (await svc.get_structure("u1", "2026-08-10"))["sleep_structure"]
    assert structure["deep_sleep"] == {"minutes": 108, "percent": 20}
    quality = (await svc.get_sleep_quality("u1", "2026-08-10"))["sleep_quality"]
    assert quality["time_in_bed_minutes"] == 540
    assert quality["bedtime"] == "23:30"
    assert quality["wake_up_time"] == "08:30"
    assert quality["awake_after_onset_minutes"] == 18
    assert quality["sleep_onset_latency_minutes"] == 20


@pytest.mark.asyncio
async def test_mongo_not_configured() -> None:
    svc = ReportService(client=None, settings=_settings())
    with pytest.raises(MongoNotConfiguredError):
        await svc.get_summary("u1", "2026-08-10")
