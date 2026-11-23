"""量产 ReportService 业务单测。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import MongoNotConfiguredError
from app.server.somni.report import calc
from app.server.somni.report.service import ReportService


def _settings() -> Settings:
    return Settings(somni_mongo_uri="mongodb://localhost")


def _service(store: MagicMock) -> ReportService:
    """补齐异步默认桩：真实 Store 里 by_session 会转到 by_filter。"""
    if not isinstance(store.find_session_payload, AsyncMock):
        store.find_session_payload = AsyncMock(return_value=None)
    if not isinstance(store.list_telemetry_by_session, AsyncMock):

        async def _by_session(session_id: str, metric: str | None = None):
            sid = (session_id or "").strip()
            if not sid:
                return []
            if isinstance(store.list_telemetry_by_filter, AsyncMock):
                query: dict = {"session_id": sid}
                if metric:
                    query["metric"] = metric
                return await store.list_telemetry_by_filter(query)
            if isinstance(store.list_telemetry, AsyncMock):
                return await store.list_telemetry()
            return []

        store.list_telemetry_by_session = AsyncMock(side_effect=_by_session)
    return ReportService(client=MagicMock(), settings=_settings(), store=store)


def _session_with_stages() -> dict:
    """吸气约 9h 在床、含 deep 段，便于 summary/structure 断言。"""
    return {
        "timezone": "Asia/Shanghai",
        "payload": {
            "on_bed": [
                {
                    "from": int(datetime(2026, 8, 9, 15, 30, tzinfo=UTC).timestamp() * 1000),
                    "to": int(datetime(2026, 8, 10, 0, 30, tzinfo=UTC).timestamp() * 1000),
                }
            ],
            "final_stages": {
                "segments": [
                    {
                        "stage": 0,
                        "stage_name": "Awake",
                        "start_ts_ms": int(
                            datetime(2026, 8, 9, 15, 30, tzinfo=UTC).timestamp() * 1000
                        ),
                        "duration_ms": 54 * 60 * 1000,
                    },
                    {
                        "stage": 3,
                        "stage_name": "Deep",
                        "start_ts_ms": int(
                            datetime(2026, 8, 9, 16, 24, tzinfo=UTC).timestamp() * 1000
                        ),
                        "duration_ms": 108 * 60 * 1000,
                    },
                    {
                        "stage": 2,
                        "stage_name": "Light",
                        "start_ts_ms": int(
                            datetime(2026, 8, 9, 18, 12, tzinfo=UTC).timestamp() * 1000
                        ),
                        "duration_ms": 270 * 60 * 1000,
                    },
                    {
                        "stage": 1,
                        "stage_name": "REM",
                        "start_ts_ms": int(
                            datetime(2026, 8, 9, 22, 42, tzinfo=UTC).timestamp() * 1000
                        ),
                        "duration_ms": 108 * 60 * 1000,
                    },
                ]
            },
        },
    }


@pytest.mark.asyncio
async def test_get_summary_computes_stages_and_avg() -> None:
    store = MagicMock()
    store.find_session_payload = AsyncMock(return_value=_session_with_stages())
    store.list_telemetry = AsyncMock(
        return_value=[
            {"data": {"hr": 66.5, "br": 16}},
            {"data": {"hr": 62.2, "br": 14.8}},
        ]
    )
    summary = (await _service(store).get_summary("u1", "2026-08-10", "sess-1"))[
        "sleep_summary"
    ]
    assert summary["body_battery"] == 0
    assert summary["body_battery_status"] == ""
    assert summary["deep_sleep_minutes"] == 108
    assert summary["total_minutes"] >= 108
    assert summary["avg_heart_rate"] == 64
    assert summary["avg_respiratory_rate"] == 15


@pytest.mark.asyncio
async def test_get_summary_without_session_zeros() -> None:
    store = MagicMock()
    store.find_session_payload = AsyncMock(return_value=None)
    store.list_telemetry_by_session = AsyncMock(return_value=[])
    summary = (await _service(store).get_summary("u1", "2026-08-10", "sess-1"))[
        "sleep_summary"
    ]
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
    store.list_telemetry = AsyncMock(
        return_value=[
            {"data": {"temp": 21.9, "humi": 40.2, "lux": 1.2, "noise_db": 28.9}},
            {"data": {"temp": 25.8, "humi": 55.1, "lux": 4.9, "noise_db": 34.1}},
        ]
    )
    env = (
        await _service(store).get_environment("u1", "2026-08-10", "sess-1")
    )["environment_summary"]
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
    store.find_session_payload = AsyncMock(
        return_value={
            "timezone": "Asia/Shanghai",
            "payload": {
                "final_stages": {
                    "segments": [
                        {
                            "stage": 3,
                            "stage_name": "Deep",
                            "start_ts_ms": 1_754_767_200_000,
                            "duration_ms": 4_800_000,
                        }
                    ]
                }
            },
        }
    )
    store.list_telemetry_by_filter = AsyncMock(
        side_effect=lambda query: {
            "sleep": [
                {
                    "ts": datetime(2026, 8, 9, 18, 15, tzinfo=UTC),
                    "data": {"hr": 68.2, "br": 16.1},
                }
            ],
            "env": [
                {
                    "ts": datetime(2026, 8, 9, 18, 15, tzinfo=UTC),
                    "data": {"temp": 23.2, "humi": 52.9, "lux": 1.2, "noise_db": 28.4},
                }
            ],
        }.get(query.get("metric"), [])
    )
    payload = await _service(store).get_events("u1", "2026-08-10", "sess-1")
    assert payload["event_count"] == 1
    assert payload["abnormal_count"] == 1
    assert payload["intervention_count"] == 1
    assert len(payload["sleep_events"]) == 1
    assert payload["sleep_events"][0]["intervention"]["event_type"] == "干预"
    assert payload["idf_data"][0]["stage"] == "deep"
    assert payload["physio_data"][0]["metrics"]["heart_rate"] == 68
    assert payload["env_data"][0]["noise"] == 28
    queries = [call.args[0] for call in store.list_telemetry_by_filter.await_args_list]
    assert {"session_id": "sess-1", "metric": "sleep"} in queries
    assert {"session_id": "sess-1", "metric": "env"} in queries
    store.list_telemetry.assert_not_called()


@pytest.mark.asyncio
async def test_get_events_percent_bucket_physio_and_half_hour_env() -> None:
    """physio 仍按 10% 分桶；env 按半小时分桶（锚点最早采样）。"""
    store = MagicMock()
    store.list_events = AsyncMock(return_value=[])
    store.find_session_payload = AsyncMock(
        return_value={"timezone": "Asia/Shanghai", "payload": {}}
    )
    anchor = datetime(2026, 8, 9, 18, 0, tzinfo=UTC)
    store.list_telemetry_by_filter = AsyncMock(
        side_effect=lambda query: {
            "sleep": [
                {"ts": anchor, "data": {"hr": 60.4, "br": 14.5}},
                {
                    "ts": anchor + timedelta(minutes=5),
                    "data": {"hr": 61.4, "br": 15.5},
                },
                {
                    "ts": anchor + timedelta(minutes=50),
                    "data": {"hr": 70.2, "br": 16.2},
                },
                {
                    "ts": anchor + timedelta(minutes=100),
                    "data": {"hr": 80.0, "br": 17.0},
                },
            ],
            "env": [
                {
                    "ts": anchor,
                    "data": {"temp": 23.2, "humi": 50.4, "lux": 1.2, "noise_db": 28.5},
                },
                {
                    "ts": anchor + timedelta(minutes=5),
                    "data": {"temp": 24.2, "humi": 51.4, "lux": 1.4, "noise_db": 29.5},
                },
                {
                    "ts": anchor + timedelta(minutes=50),
                    "data": {"temp": 25.2, "humi": 52.4, "lux": 2.4, "noise_db": 31.5},
                },
                {
                    "ts": anchor + timedelta(minutes=100),
                    "data": {"temp": 22.2, "humi": 48.2, "lux": 0.4, "noise_db": 30.2},
                },
            ],
        }.get(query.get("metric"), [])
    )
    payload = await _service(store).get_events("u1", "2026-08-10", "sess-1")
    assert payload["physio_data"] == [
        {
            "collected_at": "2026-08-10T02:00:00+08:00",
            "metrics": {"heart_rate": 61, "respiration_rate": 15},
        },
        {
            "collected_at": "2026-08-10T02:50:00+08:00",
            "metrics": {"heart_rate": 70, "respiration_rate": 16},
        },
        {
            "collected_at": "2026-08-10T03:30:00+08:00",
            "metrics": {"heart_rate": 80, "respiration_rate": 17},
        },
    ]
    # env：0/5min → 02:00；50min → 02:30；100min → 03:30
    assert payload["env_data"] == [
        {
            "collected_at": "2026-08-10T02:00:00+08:00",
            "temperature": 24,
            "humidity": 51,
            "illuminance": 1,
            "noise": 29,
        },
        {
            "collected_at": "2026-08-10T02:30:00+08:00",
            "temperature": 25,
            "humidity": 52,
            "illuminance": 2,
            "noise": 32,
        },
        {
            "collected_at": "2026-08-10T03:30:00+08:00",
            "temperature": 22,
            "humidity": 48,
            "illuminance": 0,
            "noise": 30,
        },
    ]


@pytest.mark.asyncio
async def test_get_events_env_physio_use_session_timezone() -> None:
    store = MagicMock()
    store.list_events = AsyncMock(return_value=[])
    store.find_session_payload = AsyncMock(
        return_value={"timezone": "America/Los_Angeles", "payload": {}}
    )
    anchor = datetime(2026, 8, 9, 18, 0, tzinfo=UTC)
    store.list_telemetry_by_filter = AsyncMock(
        side_effect=lambda query: {
            "sleep": [{"ts": anchor, "data": {"hr": 60, "br": 14}}],
            "env": [
                {
                    "ts": anchor,
                    "data": {"temp": 23.0, "humi": 50.0, "lux": 1.0, "noise_db": 28.0},
                }
            ],
        }.get(query.get("metric"), [])
    )
    payload = await _service(store).get_events("u1", "2026-08-10", "sess-1")
    assert payload["physio_data"][0]["collected_at"] == "2026-08-09T11:00:00-07:00"
    assert payload["env_data"][0]["collected_at"] == "2026-08-09T11:00:00-07:00"


@pytest.mark.asyncio
async def test_get_structure_and_sleep_quality() -> None:
    store = MagicMock()
    store.find_session_payload = AsyncMock(return_value=_session_with_stages())
    store.find_user_timezone = AsyncMock(return_value="Asia/Shanghai")
    store.list_records = AsyncMock(
        return_value=[
            {
                "record_date": "2026-08-10",
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
                },
            }
        ]
    )
    svc = _service(store)
    payload = await svc.get_structure(
        "u1",
        record_date="2026-08-10",
        start_date="2026-08-10",
        end_date="2026-08-10",
        session_id="sess-1",
    )
    structure = payload["sleep_structure"]
    assert structure["deep_sleep"]["minutes"] == 108
    weekly = payload["weekly_report"]
    assert weekly["timezone"] == "Asia/Shanghai"
    assert weekly["start_date"] == "2026-08-10"
    assert weekly["end_date"] == "2026-08-10"
    assert weekly["daytime_activity"] == {}
    assert len(weekly["week_days"]) == 1
    assert len(weekly["baseline_days"]) == 14
    assert weekly["week_days"][0]["record_date"] == "2026-08-10"
    store.list_records.assert_awaited_once_with("u1", "2026-07-28", "2026-08-10")

    quality = (
        await svc.get_sleep_quality("u1", "2026-08-10", "sess-1")
    )["sleep_quality"]
    assert quality["time_in_bed_minutes"] == 540
    assert quality["bedtime"] == "23:30"
    assert quality["wake_up_time"] == "08:30"


@pytest.mark.asyncio
async def test_get_structure_defaults_trailing_week_when_range_omitted() -> None:
    store = MagicMock()
    store.find_user_timezone = AsyncMock(return_value="")
    store.list_records = AsyncMock(return_value=[])
    svc = _service(store)
    payload = await svc.get_structure("u1")
    weekly = payload["weekly_report"]
    assert weekly["timezone"] == "Asia/Shanghai"
    assert len(weekly["week_days"]) == 7
    assert len(weekly["baseline_days"]) == 14
    assert all(d["total_sleep_minutes"] == 0 for d in weekly["week_days"])
    assert payload["sleep_structure"]["awake"] == {"minutes": 0, "percent": 0}
    _args = store.list_records.await_args.args
    assert _args[0] == "u1"
    assert weekly["start_date"] <= weekly["end_date"]
    assert _args[2] == weekly["end_date"]


@pytest.mark.asyncio
async def test_get_structure_record_date_only_does_not_anchor_week() -> None:
    store = MagicMock()
    store.find_user_timezone = AsyncMock(return_value="Asia/Shanghai")
    store.list_records = AsyncMock(return_value=[])
    svc = _service(store)
    payload = await svc.get_structure("u1", record_date="2026-09-28")
    weekly = payload["weekly_report"]
    wake_start, wake_end = calc.default_week_bounds(timezone_name="Asia/Shanghai")
    assert weekly["start_date"] == wake_start
    assert weekly["end_date"] == wake_end
    assert len(weekly["week_days"]) == 7
    assert len(weekly["baseline_days"]) == 14
    assert weekly["week_days"][0]["record_date"] == weekly["start_date"]
    assert weekly["week_days"][-1]["record_date"] == weekly["end_date"]
    store.list_records.assert_awaited_once_with(
        "u1", calc.baseline_start(wake_end), wake_end
    )


@pytest.mark.asyncio
async def test_get_structure_invalid_timezone_falls_back_and_echoes() -> None:
    store = MagicMock()
    store.find_user_timezone = AsyncMock(return_value="Not/AZone")
    store.list_records = AsyncMock(return_value=[])
    svc = _service(store)
    weekly = (await svc.get_structure("u1", record_date="2026-09-28"))["weekly_report"]
    assert weekly["timezone"] == "Asia/Shanghai"


@pytest.mark.asyncio
async def test_get_structure_tolerates_iso_string_bed_times() -> None:
    store = MagicMock()
    record = {
        "record_date": "2026-08-10",
        "raw_data": {
            "bed_time": "2026-08-09T15:30:00+00:00",
            "wake_up_time": "2026-08-10T00:30:00+00:00",
            "awake_ratio": 10,
            "deep_sleep_ratio": 20,
            "light_sleep_ratio": 50,
            "rem_ratio": 20,
            "sleep_efficiency": 92,
        },
    }
    store.find_session_payload = AsyncMock(return_value=_session_with_stages())
    store.find_user_timezone = AsyncMock(return_value="Asia/Shanghai")
    store.list_records = AsyncMock(return_value=[record])
    svc = _service(store)
    payload = await svc.get_structure(
        "u1",
        record_date="2026-08-10",
        start_date="2026-08-10",
        end_date="2026-08-10",
        session_id="sess-1",
    )
    day = payload["weekly_report"]["week_days"][0]
    assert day["bed_time"] == "2026-08-09T23:30:00+08:00"
    assert day["wake_up_time"] == "2026-08-10T08:30:00+08:00"
    assert day["total_sleep_minutes"] == 486
    assert payload["sleep_structure"]["deep_sleep"]["minutes"] == 108


@pytest.mark.asyncio
async def test_mongo_not_configured() -> None:
    svc = ReportService(client=None, settings=_settings())
    with pytest.raises(MongoNotConfiguredError):
        await svc.get_summary("u1", "2026-08-10", "sess-1")
