from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.server.somni.facts.models import DataError
from app.server.somni.facts.service import GenerationDataService
from app.server.somni.facts.sleep_debt import calculate_sleep_debt

CUTOFF = datetime(2026, 9, 21, 12, tzinfo=UTC)


def night(day: str, minutes: float = 480, **changes) -> dict:
    return {
        "record_date": day,
        "episode_id": day,
        "revision": 1,
        "source": "wearable",
        "timezone_offset": "+00:00",
        "device_algorithm": "v1",
        "definition_version": "v1",
        "main_sleep": True,
        "selection_policy_version": "v1",
        "quality": {"valid": True},
        "received_at": f"{day}T09:00:00Z",
        "sleep": {
            "total_asleep_min": minutes,
            "start_at": f"{day}T00:00:00Z",
            "end_at": f"{day}T08:00:00Z",
        },
        **changes,
    }


@pytest.mark.parametrize(
    "count,target,debt", [(0, 450, 30), (13, 450, 30), (14, 480, 60), (16, 480, 60)]
)
def test_prior_valid_days_not_calendar_days(count, target, debt):
    rows = [
        night((date(2026, 9, 21) - timedelta(days=3 * index)).isoformat())
        for index in range(1, count + 1)
    ]
    result = calculate_sleep_debt(rows + [night("2026-09-21", 420)], "2026-09-21", CUTOFF)
    assert result["value"]["target_sleep_min"] == target
    assert result["value"]["daily_debt_min"] == debt
    assert result["value"]["baseline_valid_day_count"] == min(14, count)


def test_missing_current_and_surplus():
    assert "daily_debt_min" not in calculate_sleep_debt([], "2026-09-21", CUTOFF)["value"]
    assert (
        calculate_sleep_debt([night("2026-09-21", 500)], "2026-09-21", CUTOFF)["value"][
            "daily_debt_min"
        ]
        == 0
    )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_measurements(value):
    assert (
        "daily_debt_min"
        not in calculate_sleep_debt([night("2026-09-21", value)], "2026-09-21", CUTOFF)["value"]
    )


def test_revision_and_source_boundaries():
    rows = [
        night("2026-09-21", 300),
        night("2026-09-21", 420, revision=2),
        night("2026-09-20", 600, source="radar"),
        night("2026-09-22", 600),
    ]
    result = calculate_sleep_debt(rows, "2026-09-21", CUTOFF)["value"]
    assert result["daily_debt_min"] == 30 and result["baseline_valid_day_count"] == 0
    assert (
        calculate_sleep_debt([night("2026-09-21", 0)], "2026-09-21", CUTOFF)["value"][
            "daily_debt_min"
        ]
        == 450
    )


def test_revision_retracts_main_sleep_and_naps_do_not_duplicate():
    current = night("2026-09-21", 420)
    retracted = night("2026-09-21", 420, revision=2, main_sleep=False)
    assert (
        "daily_debt_min"
        not in calculate_sleep_debt([current, retracted], "2026-09-21", CUTOFF)["value"]
    )
    nap = night("2026-09-21", 30, episode_id="nap", main_sleep=False)
    assert (
        calculate_sleep_debt([current, nap], "2026-09-21", CUTOFF)["value"]["daily_debt_min"] == 30
    )


async def test_profile_debt_only_and_reminder_dates(monkeypatch):
    monkeypatch.setattr("app.server.somni.facts.service.utc_now", lambda: CUTOFF)
    store = AsyncMock()
    store.load_permissions.return_value = {
        "status": "active",
        "privacy_switches": {"health": True, "calendar": True},
    }
    store.load_sleep_debt_rows.return_value = [night("2026-09-21", 420)]
    store.load_reminder_schedules.return_value = []
    svc = GenerationDataService(None, Settings(_env_file=None), store)
    profile = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="initial",
        review_end_local_date="2026-09-22",
        timezone_offset="+08:00",
        requested_sections=["sleep_debt"],
        data_cutoff_at_utc="2026-09-21T12:00:00Z",
    )
    assert set(profile["data"]) == {"sleep_debt"}
    assert profile["data"]["sleep_debt"]["value"]["daily_debt_min"] == 30
    store.load_questionnaires.assert_not_called()
    response = await svc.get_bedtime_reminder_inputs(
        "u", "r", reminder_local_date="2026-09-21", timezone_offset="+08:00"
    )
    assert response["target_local_date"] == "2026-09-22"
    assert response["data"]["weather"]["status"] == "unsupported"
    assert response["data"]["sleep_debt"]["value"]["record_date"] == "2026-09-21"


async def test_source_failure_is_not_fallback_and_reminder_schedules_are_queried():
    store = AsyncMock()
    store.load_permissions.return_value = {"status": "active", "privacy_switches": {"health": True}}
    store.load_sleep_debt_rows.side_effect = DataError("UNAVAILABLE", "source_unavailable")
    svc = GenerationDataService(None, Settings(_env_file=None), store)
    result = await svc.get_bedtime_reminder_inputs(
        "u", "r", reminder_local_date="2026-09-21", timezone_offset="+05:30"
    )
    assert result["data"]["sleep_debt"]["status"] == "unavailable"
    assert "value" not in result["data"]["sleep_debt"]
    store.load_reminder_schedules.assert_awaited_once()
    store.load_permissions.assert_not_awaited()


@pytest.mark.parametrize(
    "changes",
    [
        {"received_at": "2026-09-22T00:00:00Z"},
        {"revision": True},
        {"timezone_offset": "invalid"},
        {"quality": {"valid": False}},
    ],
)
def test_missing_or_future_provenance_is_unknown(changes):
    result = calculate_sleep_debt([night("2026-09-21", **changes)], "2026-09-21", CUTOFF)
    assert "daily_debt_min" not in result["value"]


async def test_calendar_interval_cancellation_and_all_day():
    store = AsyncMock()
    store.load_reminder_schedules.return_value = [
        {"event_id": "all", "event_date": "2026-09-20", "end_date": "2026-09-24", "all_day": True},
        {
            "event_id": "overnight",
            "event_date": "2026-09-21",
            "start_at": "2026-09-21T15:00:00Z",
            "end_at": "2026-09-21T17:00:00Z",
        },
        {"event_id": "cancel", "event_date": "2026-09-22", "all_day": True, "status": "cancelled"},
        {
            "event_id": "ambiguous",
            "event_date": "2026-09-22",
            "start_time": "08:00",
            "end_time": "09:00",
        },
    ]
    result = await GenerationDataService(None, Settings(_env_file=None), store)._reminder_schedules(
        "u", "2026-09-22", "+08:00", CUTOFF, {}
    )
    assert [item["event_id"] for item in result["value"]] == ["all", "overnight"]
    assert "schedule_timezone_missing" in result["missing_reasons"]
    assert result["status"] == "partial"
