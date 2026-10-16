import hashlib
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.server.somni.facts.models import DataError, canonical_hash, timestamp, utc_now
from app.server.somni.facts.service import GenerationDataService


@pytest.mark.parametrize("schema_version", ["0.0.1", "2"])
@pytest.mark.parametrize(
    "body,reason",
    [
        ({}, None),
        (None, "incomplete_profile_content"),
        ({"value": float("nan")}, "invalid_canonical_content"),
    ],
)
async def test_original_profile_hash_is_metadata_with_finite_body_required(
    monkeypatch, schema_version, body, reason
) -> None:
    from app.server.somni.facts.models import compute_long_term_content_hash

    svc, store = service()
    short = profile()
    short.update(
        schema_version=schema_version,
        snapshot_id="snapshot-1",
        long_term_content_hash="sha256:" + "a" * 64,
    )
    long = {
        "uid": "u",
        "schema_version": schema_version,
        "status": "valid",
        "content_hash": short["long_term_content_hash"],
        "long_term_profile": body,
    }
    store.load_profile.return_value = (short, long)
    calls = []

    def envelope_hash(value):
        calls.append(value)
        return canonical_hash(value)

    monkeypatch.setattr("app.server.somni.facts.service.canonical_hash", envelope_hash)
    rejected = await svc._profile("u", "", utc_now(), {})
    if reason is None:
        assert rejected["status"] == "ok"
        assert rejected["value"]["long_term_content_hash"] == short["long_term_content_hash"]
    else:
        assert rejected["status"] == "partial"
        assert rejected["missing_reasons"] == [reason]
        assert "value" not in rejected

    calls.clear()
    long["long_term_profile"] = {}
    short["long_term_content_hash"] = compute_long_term_content_hash({}, schema_version)
    long["content_hash"] = short["long_term_content_hash"]
    accepted = await svc._profile("u", "", utc_now(), {})
    assert accepted["status"] == "ok"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "purpose", ["plan_generation", "profile_refresh", "initial", "weekly", "correction"]
)
@pytest.mark.parametrize("schema_version", ["0.0.1", "2"])
async def test_all_profile_purposes_skip_original_profile_hash(purpose, schema_version) -> None:
    svc, store = service()
    short = profile()
    short.update(schema_version=schema_version, snapshot_id="test-placeholder", snapshot_revision=1)
    long = {
        "uid": "u",
        "schema_version": schema_version,
        "status": "valid",
        "content_hash": short["long_term_content_hash"],
        "long_term_profile": {"test-placeholder": "changed-body"},
    }
    original = deepcopy((short, long))
    store.load_profile.return_value = (short, long)
    result = await svc.get_profile_inputs(
        "u",
        "test-placeholder",
        purpose=purpose,
        review_end_local_date="2026-09-21",
        timezone_offset="+00:00",
        review_generation_id="test-placeholder" if purpose == "weekly" else "",
        requested_sections=["prior_profile"],
    )
    received = result["data"]["prior_profile"]
    assert received["status"] == "ok"
    assert received["missing_reasons"] == ["long_term_hash_verification_skipped"]
    assert received["value"]["long_term_profile"] == long["long_term_profile"]
    assert received["value"]["long_term_content_hash"] == short["long_term_content_hash"]
    assert (short, long) == original


@pytest.mark.parametrize("body", [None, [], {"test-placeholder": float("nan")}])
async def test_plan_hash_bypass_still_rejects_malformed_body(body) -> None:
    svc, store = service()
    short = profile()
    store.load_profile.return_value = (
        short,
        {
            "uid": "u",
            "schema_version": short["schema_version"],
            "content_hash": short["long_term_content_hash"],
            "long_term_profile": body,
        },
    )
    result = await svc._profile("u", "", utc_now(), {}, verify_long_term_hash=False)
    assert result["status"] == "partial"
    assert "value" not in result


def profile() -> dict:
    value = {
        "uid": "u",
        "schema_version": "2",
        "record_date": "2026-09-20",
        "status": "active",
        "data_cutoff_at_utc": "2026-09-20T00:00:00Z",
        "generated_at_utc": "2026-09-20T00:00:00Z",
        "long_term_profile": {},
        "short_term_profile": {},
        "rolling_baseline": {},
        "profile_quality": {},
        "based_on": {},
        "long_term_content_hash": canonical_hash({}),
    }
    return {**value, "content_hash": canonical_hash(value)}


def service() -> tuple:
    store = AsyncMock()
    store.load_active_plan.return_value = []
    store.load_profile.return_value = (None, None)
    store.list_sleep_episodes.return_value = []
    store.load_daily_metrics.return_value = []
    store.list_daily_sleep_reports.return_value = []
    store.list_sessions.return_value = []
    store.list_execution_events.return_value = {
        "operations": [],
        "downlinks": [],
        "events": [],
        "session_payloads": [],
    }
    store.load_permissions.return_value = {
        "status": "active",
        "privacy_switches": {
            "profiles": True,
            "plans": True,
            "health": True,
            "questionnaires": True,
            "observations": True,
        },
    }
    settings = Settings(_env_file=None, generation_cursor_secret="synthetic-test-key-" * 3)
    return GenerationDataService(None, settings, store), store


async def test_build_facts_projects_daily_report_vitals_with_report_provenance() -> None:
    svc, store = service()
    store.list_sleep_episodes.return_value = []
    store.load_daily_metrics.return_value = []
    store.list_daily_sleep_reports.return_value = [
        {
            "uid": "u",
            "reports_id": "report-20261007",
            "record_date": "2026-10-07",
            "language": "en",
            "overnight_vitals": {
                "hr": {"value": 61.5},
                "br": {"value": 14.25},
            },
        }
    ]
    from app.server.somni.facts.time import resolve_scope

    manifest = {}
    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+00:00"), datetime(2026, 10, 8, tzinfo=UTC), manifest
    )

    assert facts["status"] == "partial"
    series = {item["metric_id"]: item for item in facts["value"]["series"]}
    assert series["sleep_hr_mean_bpm"]["daily"][0]["value"] == 61.5
    assert series["sleep_resp_mean_per_min"]["daily"][0]["value"] == 14.25
    assert series["sleep_hr_mean_bpm"]["daily"][0]["source_refs"] == [
        "sleep_reports:report-20261007"
    ]
    assert manifest["sleep_reports:report-20261007"]["record_id"] == "report-20261007"


async def test_build_facts_preserves_other_sources_and_exclusion_reason() -> None:
    from app.server.somni.facts.time import resolve_scope

    svc, store = service()

    async def exclude_late_record(*args, missing_reasons):
        missing_reasons.append("records_historical_version_unavailable")
        return []

    store.list_sleep_episodes.side_effect = exclude_late_record
    store.list_daily_sleep_reports.return_value = [
        {
            "uid": "u",
            "reports_id": "synthetic-report",
            "record_date": "2026-10-07",
            "language": "en",
            "overnight_vitals": {"hr": {"value": 61.5}},
        }
    ]
    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+00:00"), datetime(2026, 10, 8, tzinfo=UTC), {}
    )
    assert facts["status"] == "partial"
    assert "records_historical_version_unavailable" in facts["missing_reasons"]
    assert facts["value"]["series"][0]["daily"][0]["value"] == 61.5


async def test_build_facts_deduplicates_equal_language_report_values() -> None:
    svc, store = service()
    store.list_daily_sleep_reports.return_value = [
        {
            "uid": "u",
            "reports_id": "report-en",
            "record_date": "2026-10-07",
            "language": "en",
            "overnight_vitals": {"hr": {"value": 60}, "br": {"value": 14}},
        },
        {
            "uid": "u",
            "reports_id": "report-zh",
            "record_date": "2026-10-07",
            "language": "zh",
            "overnight_vitals": {"hr": {"value": 60.0}, "br": {"value": 14.0}},
        },
    ]
    from app.server.somni.facts.time import resolve_scope

    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+00:00"), datetime(2026, 10, 8, tzinfo=UTC), {}
    )

    hr = next(item for item in facts["value"]["series"] if item["metric_id"] == "sleep_hr_mean_bpm")
    assert hr["stats_28d"]["valid_count"] == 1
    assert len(hr["daily"][0]["source_refs"]) == 2


async def test_build_facts_omits_conflicting_language_report_values() -> None:
    svc, store = service()
    store.list_daily_sleep_reports.return_value = [
        {
            "uid": "u",
            "reports_id": "report-en",
            "record_date": "2026-10-07",
            "language": "en",
            "overnight_vitals": {"hr": {"value": 60}, "br": {"value": 14}},
        },
        {
            "uid": "u",
            "reports_id": "report-zh",
            "record_date": "2026-10-07",
            "language": "zh",
            "overnight_vitals": {"hr": {"value": 72}, "br": {"value": 14}},
        },
    ]
    from app.server.somni.facts.time import resolve_scope

    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+00:00"), datetime(2026, 10, 8, tzinfo=UTC), {}
    )

    assert not any(item["metric_id"] == "sleep_hr_mean_bpm" for item in facts["value"]["series"])
    assert any(
        reason.startswith("sleep_report_language_conflict:sleep_hr_mean_bpm:")
        for reason in facts["missing_reasons"]
    )


async def test_build_facts_prefers_usable_daily_fact_over_report_projection() -> None:
    svc, store = service()
    store.list_sleep_episodes.return_value = [
        {
            "_id": "daily-fact-1",
            "episode_id": "episode-1",
            "source": "wearable",
            "record_date": "2026-10-07",
            "revision": 1,
            "main_sleep": True,
            "selection_policy_version": "main_sleep_v1",
            "device_algorithm": "wearable_v1",
            "definition_version": "metrics_v1",
            "quality": {"valid": True},
            "sleep": {
                "start_at": "2026-10-06T22:00:00Z",
                "end_at": "2026-10-07T06:00:00Z",
            },
            "physiology": {
                "heart_rate_sleep_avg_bpm": 58,
                "respiratory_rate_sleep_avg": 13,
            },
        },
    ]
    store.list_daily_sleep_reports.return_value = [
        {
            "uid": "u",
            "reports_id": "report-20261007",
            "record_date": "2026-10-07",
            "language": "en",
            "overnight_vitals": {"hr": {"value": 70}, "br": {"value": 17}},
        }
    ]
    from app.server.somni.facts.time import resolve_scope

    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+00:00"), datetime(2026, 10, 8, tzinfo=UTC), {}
    )

    series = {item["metric_id"]: item for item in facts["value"]["series"]}
    assert series["sleep_hr_mean_bpm"]["source"] == "wearable"
    assert series["sleep_hr_mean_bpm"]["daily"][0]["value"] == 58
    assert series["sleep_resp_mean_per_min"]["daily"][0]["value"] == 13


async def test_build_facts_fills_report_gaps_per_metric_and_local_date() -> None:
    svc, store = service()
    store.list_sleep_episodes.return_value = [
        {
            "_id": "daily-fact-20261006",
            "source": "wearable",
            "record_date": "2026-10-06",
            "revision": 1,
            "main_sleep": True,
            "selection_policy_version": "main_sleep_v1",
            "device_algorithm": "wearable_v1",
            "definition_version": "metrics_v1",
            "quality": {"valid": True},
            "sleep": {
                "start_at": "2026-10-05T22:00:00Z",
                "end_at": "2026-10-06T06:00:00Z",
            },
            "physiology": {
                "heart_rate_sleep_avg_bpm": 58,
                "respiratory_rate_sleep_avg": 13,
            },
        },
        {
            "_id": "invalid-fact-20261007",
            "source": "wearable",
            "record_date": "2026-10-07",
            "revision": 1,
            "main_sleep": True,
            "selection_policy_version": "main_sleep_v1",
            "device_algorithm": "wearable_v1",
            "definition_version": "metrics_v1",
            "quality": {"valid": True},
            "sleep": {
                "start_at": "2026-10-06T22:00:00Z",
                "end_at": "2026-10-07T06:00:00Z",
            },
            "physiology": {
                "heart_rate_sleep_avg_bpm": "invalid",
                "respiratory_rate_sleep_avg": True,
            },
        },
        {
            "_id": "record-fact-20261007",
            "source_type": "records",
            "source": "somni_records",
            "record_date": "2026-10-07",
            "revision": 1,
            "main_sleep": True,
            "selection_policy_version": "main_sleep_v1",
            "device_algorithm": "radar_v1",
            "definition_version": "record_metrics_v1",
            "quality": {"valid": True},
            "physiology": {
                "heart_rate_sleep_avg_bpm": 55,
                "respiratory_rate_sleep_avg": 12,
            },
        },
    ]
    store.list_daily_sleep_reports.return_value = [
        {
            "uid": "u",
            "reports_id": "report-20261006",
            "record_date": "2026-10-06",
            "language": "en",
            "overnight_vitals": {"hr": {"value": 70}, "br": {"value": 16}},
        },
        {
            "uid": "u",
            "reports_id": "report-20261007",
            "record_date": "2026-10-07",
            "language": "en",
            "overnight_vitals": {"hr": {"value": 72}, "br": {"value": 14}},
        },
    ]
    from app.server.somni.facts.time import resolve_scope

    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+00:00"), datetime(2026, 10, 8, tzinfo=UTC), {}
    )

    series = {item["metric_id"]: item for item in facts["value"]["series"]}
    heart_rate = series["sleep_hr_mean_bpm"]
    daily = {item["local_date"]: item for item in heart_rate["daily"]}
    assert heart_rate["source"] == "mixed"
    assert heart_rate["stats_7d"]["valid_count"] == 2
    assert daily["2026-10-06"]["value"] == 58
    assert daily["2026-10-06"]["source"] == "wearable"
    assert daily["2026-10-07"]["value"] == 72
    assert daily["2026-10-07"]["source"] == "somni_sleep_reports"


async def test_build_facts_projects_somni_records_when_reports_are_absent() -> None:
    svc, store = service()
    store.list_sleep_episodes.return_value = [
        {
            "_id": "record-20261007",
            "source_type": "records",
            "source": "somni_records",
            "record_date": "2026-10-07",
            "main_sleep": True,
            "timezone_offset": "+08:00",
            "raw_data": {
                "bed_time": datetime(2026, 10, 6, 14, tzinfo=UTC),
                "wake_time": datetime(2026, 10, 6, 20, tzinfo=UTC),
                "wake_up_time": datetime(2026, 10, 6, 22, tzinfo=UTC),
                "sleep_latency": 20,
                "sleep_efficiency": 85,
                "awake_ratio": 5,
                "rem_ratio": 20,
                "light_sleep_ratio": 50,
                "deep_sleep_ratio": 25,
            },
        }
    ]
    store.list_daily_sleep_reports.return_value = []
    from app.server.somni.facts.time import resolve_scope

    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+08:00"), datetime(2026, 10, 8, tzinfo=UTC), {}
    )

    series = {item["metric_id"]: item for item in facts["value"]["series"]}
    expected = {
        "total_sleep_min": 456,
        "sol_min": 20,
        "waso_min": 120,
        "sleep_efficiency_pct": 85,
    }
    for metric_id, value in expected.items():
        daily = series[metric_id]["daily"]
        assert len(daily) == 1
        assert daily[0]["local_date"] == "2026-10-07"
        assert daily[0]["quality"] == "valid"
        assert daily[0]["value"] == value
        assert daily[0]["source_refs"] == ["records:record-20261007"]


async def test_build_facts_rejects_ambiguous_same_date_somni_records() -> None:
    svc, store = service()
    record = {
        "_id": "record-20261007-a",
        "source_type": "records",
        "source": "somni_records",
        "record_date": "2026-10-07",
        "main_sleep": True,
        "timezone_offset": "+08:00",
        "raw_data": {
            "bed_time": datetime(2026, 10, 6, 14, tzinfo=UTC),
            "wake_time": datetime(2026, 10, 6, 20, tzinfo=UTC),
            "wake_up_time": datetime(2026, 10, 6, 22, tzinfo=UTC),
            "sleep_latency": 20,
            "sleep_efficiency": 85,
            "awake_ratio": 5,
            "rem_ratio": 20,
            "light_sleep_ratio": 50,
            "deep_sleep_ratio": 25,
        },
    }
    store.list_sleep_episodes.return_value = [
        record,
        {**record, "_id": "record-20261007-b"},
    ]
    store.list_daily_sleep_reports.return_value = []
    from app.server.somni.facts.time import resolve_scope

    facts = await svc.build_facts(
        "u", resolve_scope("2026-10-08", "+08:00"), datetime(2026, 10, 8, tzinfo=UTC), {}
    )

    assert not any(item["metric_id"] == "total_sleep_min" for item in facts["value"]["series"])
    assert "somni_record_revision_ambiguous" in facts["missing_reasons"]


async def test_plan_inline_no_health_queries() -> None:
    svc, store = service()
    result = await svc.get_plan_inputs("u", "req", inline_profile=profile())
    assert result["data"]["profile"]["status"] == "ok"
    assert result["data"]["previous_blueprint"]["status"] == "not_found"
    store.load_permissions.assert_not_awaited()
    assert store.mock_calls == []
    wrong = profile()
    wrong["uid"] = "other"
    with pytest.raises(DataError, match="owner_mismatch"):
        await svc.get_plan_inputs("u", "req", inline_profile=wrong)


async def test_previous_blueprint_is_loaded_by_exact_generation_id() -> None:
    svc, store = service()
    store.load_blueprint.return_value = {
        "uid": "u",
        "generation_id": "weekly-generation-1",
        "status": "committed",
        "raw_plan": {"uid": "u", "revision": 1},
        "decision_record": {"decision_id": "decision-1"},
        "output_plans": [],
    }
    result = await svc.get_plan_inputs(
        "u",
        "blueprint-request",
        inline_profile=profile(),
        previous_generation_id="weekly-generation-1",
    )
    blueprint = result["data"]["previous_blueprint"]
    assert blueprint["value"]["generation_id"] == "weekly-generation-1"
    store.load_blueprint.assert_awaited_once()


@pytest.mark.parametrize(
    "source,field,value,reason",
    [
        ("", "", None, ""),
        ("pointer", "date", None, "active_plan_date_mismatch"),
        ("pointer", "version", True, "active_plan_link_incomplete"),
        ("generation", "status", None, "active_plan_generation_not_committed"),
        ("generation", "status", "pending", "active_plan_generation_not_committed"),
        ("generation", "generation_id", "other", "active_plan_generation_mismatch"),
        ("snapshot", "version", None, "active_plan_snapshot_version_mismatch"),
        ("snapshot", "version", True, "active_plan_snapshot_version_mismatch"),
        ("snapshot", "version", 3, "active_plan_snapshot_version_mismatch"),
        ("snapshot", "status", None, "active_plan_snapshot_not_active"),
        ("snapshot", "status", "disabled", "active_plan_snapshot_not_active"),
        ("snapshot", "plan_start_local_date", None, "active_plan_date_mismatch"),
        ("snapshot", "plan_start_local_date", "2026-09-24", "active_plan_date_mismatch"),
    ],
)
async def test_target_plan_ids_are_resolved_from_active_pointer(
    source, field, value, reason
) -> None:
    svc, store = service()
    store.load_active_plan.return_value = [
        {
            "uid": "u",
            "date": "2026-09-23",
            "plan_id": "plan-from-middleware",
            "version": 2,
            "weekly_generation_id": "week-from-middleware",
            "generation_id": "daily-generation",
        }
    ]
    store.load_blueprint.return_value = {
        "uid": "u",
        "generation_id": "week-from-middleware",
        "status": "committed",
        "raw_plan": {"uid": "u"},
        "decision_record": {},
        "output_plans": [{"plan_id": "plan-from-middleware", "date": "2026-09-23"}],
    }
    store.load_plan_revision.return_value = {
        "uid": "u",
        "plan_id": "plan-from-middleware",
        "version": 2,
        "snapshot_id": "snapshot-2",
        "content_hash": "hash-2",
        "status": "active",
        "plan_start_local_date": "2026-09-23",
        "phases": [],
    }
    if source:
        records = {
            "pointer": store.load_active_plan.return_value[0],
            "generation": store.load_blueprint.return_value,
            "snapshot": store.load_plan_revision.return_value,
        }
        records[source][field] = value
        with pytest.raises(DataError, match=reason):
            await svc.get_plan_inputs(
                "u", "request", inline_profile=profile(), target_local_date="2026-09-23"
            )
        return
    result = await svc.get_plan_inputs(
        "u", "request", inline_profile=profile(), target_local_date="2026-09-23"
    )
    target = result["data"]["target_plan"]["value"]
    assert target["weekly_generation_id"] == "week-from-middleware"
    assert target["plan_id"] == "plan-from-middleware"
    assert target["version"] == 2
    target_only = await svc.get_plan_inputs(
        "u", "request-target-only", target_local_date="2026-09-23"
    )
    assert target_only["data"]["target_plan"]["value"]["plan_id"] == "plan-from-middleware"
    assert target_only["data"]["profile"]["status"] == "not_found"
    assert store.load_profile.await_args.args[3] == "2026-09-23"


async def test_weekly_scope_resolves_previous_blueprint_from_effective_plan() -> None:
    svc, store = service()
    store.load_latest_active_plan_date_before.return_value = "2026-10-04"
    store.load_active_plan.return_value = [
        {
            "uid": "u",
            "date": "2026-10-04",
            "plan_id": "plan-20261004",
            "version": 1,
            "weekly_generation_id": "week-prior",
            "generation_id": "daily-generation",
        }
    ]
    store.load_blueprint.return_value = {
        "uid": "u",
        "generation_id": "week-prior",
        "status": "committed",
        "generated_at": "2026-10-04T02:00:00Z",
        "raw_plan": {"uid": "u"},
        "decision_record": {},
        "output_plans": [
            {
                "plan_id": f"plan-{(date(2026, 10, 4) - timedelta(days=offset)):%Y%m%d}",
                "date": (date(2026, 10, 4) - timedelta(days=offset)).isoformat(),
            }
            for offset in range(6, -1, -1)
        ],
    }
    store.load_plan_revision.return_value = {
        "uid": "u",
        "plan_id": "plan-20261004",
        "version": 1,
        "snapshot_id": "snapshot-20261004",
        "content_hash": "hash-20261004",
        "status": "active",
        "plan_start_local_date": "2026-10-04",
        "phases": [],
    }

    result = await svc.get_plan_inputs(
        "u",
        "weekly-request",
        scope="weekly",
        target_local_date="2026-10-05",
        timezone_offset="+08:00",
    )

    assert result["data"]["previous_blueprint"]["value"]["generation_id"] == "week-prior"
    store.load_latest_active_plan_date_before.assert_awaited_once_with(
        "u", "2026-10-05", store.load_latest_active_plan_date_before.await_args.args[2]
    )


async def test_weekly_scope_allows_cold_start_only_without_plan_history() -> None:
    svc, store = service()
    store.load_latest_active_plan_date_before.return_value = None
    store.has_plan_history_before.return_value = False

    result = await svc.get_plan_inputs(
        "u",
        "weekly-first",
        scope="weekly",
        target_local_date="2026-10-05",
        timezone_offset="+08:00",
    )

    assert result["data"]["previous_blueprint"]["status"] == "not_found"
    store.load_blueprint.assert_not_awaited()


async def test_weekly_scope_rejects_missing_blueprint_when_plan_history_exists() -> None:
    svc, store = service()
    store.load_latest_active_plan_date_before.return_value = None
    store.has_plan_history_before.return_value = True

    with pytest.raises(DataError, match="previous_week_blueprint_missing"):
        await svc.get_plan_inputs(
            "u",
            "weekly-gap",
            scope="weekly",
            target_local_date="2026-10-05",
            timezone_offset="+08:00",
        )

    store.load_blueprint.assert_not_awaited()


async def test_pre_unwind_realtime_allows_missing_session() -> None:
    svc, store = service()
    result = await svc.get_plan_inputs(
        "u",
        "prebed-request",
        scope="realtime",
        timezone_offset="+08:00",
        target_local_date="2026-10-08",
        trigger_type="pre_unwind_health",
        event_id="health-delta-1",
    )
    assert result["data"]["execution_state"]["status"] == "partial"
    assert result["data"]["execution_state"]["missing_reasons"] == [
        "session_start_state_unverified"
    ]
    store.list_sessions.assert_not_awaited()


async def test_unwind_realtime_does_not_invent_execution_revision() -> None:
    svc, store = service()
    session = {
        "uid": "u",
        "session_id": "sleep-1",
        "plan_id": "plan-1",
        "plan_version": 2,
        "state": "running",
        "phase": "Unwind",
        "node_index": 0,
        "next_unexecuted_node_key": "unwind-next",
        "received_at": "2026-10-08T09:00:00Z",
    }
    store.list_sessions.return_value = [session]
    store.list_execution_events.return_value["events"] = [
        {"event_id": "anomaly-1", "type": "radar_anomaly"}
    ]
    target = {
        "status": "ok",
        "value": {
            "plan_id": "plan-1",
            "version": 2,
            "phases": [{"phase": "Unwind", "nodes": [{"node_key": "unwind-next"}]}],
        },
    }
    first = await svc._runtime_execution_state(
        "u", "sleep-1", target, utc_now(), {}, trigger_type="unwind_fusion", event_id="anomaly-1"
    )
    assert first["status"] == "partial"
    assert first["missing_reasons"] == ["authoritative_execution_boundary_unavailable"]
    assert "value" not in first

    session.pop("next_unexecuted_node_key")
    incomplete = await svc._runtime_execution_state(
        "u", "sleep-1", target, utc_now(), {}, trigger_type="unwind_fusion", event_id="anomaly-1"
    )
    assert incomplete["status"] == "partial"
    assert incomplete["missing_reasons"] == ["execution_apply_boundary_unverified"]


async def test_realtime_execution_does_not_apply_service_consent_gate() -> None:
    svc, store = service()
    store.load_permissions.return_value["privacy_switches"]["health"] = False
    svc._target_plan = AsyncMock(return_value={"status": "ok", "value": {}})
    svc._runtime_execution_state = AsyncMock(return_value={"status": "ok", "value": {}})

    result = await svc.get_plan_inputs(
        "u",
        "consent-denied",
        scope="realtime",
        trigger_type="unwind_fusion",
        event_id="health-delta",
        session_id="sleep-1",
        target_local_date="2026-10-08",
        timezone_offset="+08:00",
    )

    assert result["data"]["execution_state"]["status"] == "ok"
    svc._runtime_execution_state.assert_awaited_once()
    store.load_permissions.assert_not_awaited()


async def test_realtime_execution_cursor_requires_explicit_unexecuted_node() -> None:
    svc, store = service()
    session = {
        "uid": "u",
        "session_id": "sleep-1",
        "plan_id": "plan-1",
        "plan_version": 2,
        "state": "running",
        "phase": "Unwind",
        "node_index": 0,
        "next_unexecuted_node_key": "unwind-next",
        "received_at": "2026-10-08T09:00:00Z",
    }
    store.list_sessions.return_value = [session]
    store.list_execution_events.return_value["events"] = [
        {"event_id": "anomaly-1", "type": "radar_anomaly"}
    ]
    target_plan = {
        "status": "ok",
        "value": {
            "plan_id": "plan-1",
            "version": 2,
            "phases": [{"phase": "Unwind", "nodes": [{"node_key": "unwind-next"}]}],
        },
    }

    result = await svc._runtime_execution_state(
        "u",
        "sleep-1",
        target_plan,
        utc_now(),
        {},
        trigger_type="unwind_fusion",
        event_id="anomaly-1",
    )
    assert result["status"] == "partial"
    assert result["missing_reasons"] == ["authoritative_execution_boundary_unavailable"]
    assert "value" not in result

    del session["next_unexecuted_node_key"]
    partial = await svc._runtime_execution_state(
        "u",
        "sleep-1",
        target_plan,
        utc_now(),
        {},
        trigger_type="unwind_fusion",
        event_id="anomaly-1",
    )
    assert partial["status"] == "partial"
    assert partial["missing_reasons"] == ["execution_apply_boundary_unverified"]


async def test_profile_inputs_reads_exact_answer_ignoring_legacy_cutoff() -> None:
    svc, store = service()
    cutoff = utc_now()
    store.load_questionnaires.return_value = [
        {
            "uid": "u",
            "answer_id": "answer-exact",
            "survey_code": "initial",
            "survey_version": "v1",
            "answers": [],
        }
    ]
    await svc.get_profile_inputs(
        "u",
        "profile-request",
        purpose="initial",
        review_end_local_date="2026-09-24",
        timezone_offset="+08:00",
        requested_sections=["questionnaires"],
        answer_id="answer-exact",
        data_cutoff_at_utc=timestamp(cutoff),
    )
    uid, read_at, answer_id = store.load_questionnaires.call_args.args
    assert uid == "u" and answer_id == "answer-exact"
    assert read_at >= cutoff


@pytest.mark.parametrize("current_format", [False, True])
async def test_profile_refresh_reads_saved_pair_and_exact_sources(current_format) -> None:
    svc, store = service()
    short = profile()
    short["snapshot_id"] = "snapshot-current"
    long = {
        "uid": "u",
        "schema_version": short["schema_version"],
        "content_hash": short["long_term_content_hash"],
        "long_term_profile": short["long_term_profile"],
    }
    if current_format:
        body = {"basic_info": {"age": 30}, "fraction": 0.25}
        content_hash = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        short.update(
            schema_version="0.0.1",
            profile_snapshot_id=short.pop("snapshot_id"),
            long_term_content_hash=content_hash,
        )
        long.update(
            schema_version="0.0.1",
            content_hash=content_hash,
            long_term_profile=body,
            status="valid",
        )
    store.load_profile.return_value = (short, long)

    result = await svc.get_profile_inputs(
        "u",
        "refresh-request",
        purpose="profile_refresh",
        review_end_local_date="2026-10-09",
        timezone_offset="+08:00",
        requested_sections=["prior_profile", "questionnaires", "facts"],
    )

    assert result["data"]["prior_profile"]["status"] == "ok"
    assert result["data"]["prior_profile"]["value"]["profile_snapshot_id"] == "snapshot-current"
    assert (
        result["data"]["prior_profile"]["value"]["long_term_content_hash"]
        == short["long_term_content_hash"]
    )
    assert result["data"]["questionnaires"]["status"] == "not_found"
    assert result["data"]["facts"]["status"] == "partial"
    store.load_profile.assert_awaited_once_with(
        "u", "", store.load_profile.await_args.args[2], "2026-10-08"
    )


@pytest.mark.parametrize("selector", ["previous_profile_snapshot_id", "review_generation_id"])
async def test_profile_refresh_rejects_caller_selected_history(selector) -> None:
    svc, store = service()
    with pytest.raises(DataError, match="profile_refresh_selectors_not_allowed"):
        await svc.get_profile_inputs(
            "u",
            "refresh",
            purpose="profile_refresh",
            review_end_local_date="2026-10-09",
            timezone_offset="+08:00",
            **{selector: "caller-selected"},
        )
    store.load_permissions.assert_not_awaited()


async def test_feedback_observation_is_exact_and_date_scoped() -> None:
    svc, store = service()
    store.load_feedback_observation.return_value = {
        "uid": "u",
        "observation_id": "obs-1",
        "kind": "answer",
        "field_code": "comfort",
        "value": "warm",
        "local_date": "2026-09-23",
    }
    result = await svc.get_plan_inputs(
        "u",
        "feedback-request",
        target_local_date="2026-09-23",
        feedback_record_id="obs-1",
    )
    feedback = result["data"]["feedback"]
    assert feedback["status"] == "ok"
    assert feedback["value"][0]["observation_id"] == "obs-1"
    call_args = store.load_feedback_observation.await_args.args
    assert call_args[:2] == ("u", "obs-1")
    assert isinstance(call_args[2], datetime)


async def test_feedback_record_date_is_normalized_to_proto_local_date() -> None:
    svc, store = service()
    store.load_feedback_observation.return_value = {
        "uid": "u",
        "observation_id": "obs-record-date",
        "kind": "answer",
        "record_date": "2026-09-23",
    }
    result = await svc.get_plan_inputs(
        "u",
        "record-date-request",
        target_local_date="2026-09-23",
        feedback_record_id="obs-record-date",
    )
    assert result["data"]["feedback"]["value"][0]["local_date"] == "2026-09-23"


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (None, "feedback_record_not_found"),
        (
            {"uid": "other", "observation_id": "obs-1", "local_date": "2026-09-23"},
            "feedback_record_owner_mismatch",
        ),
        (
            {"uid": "u", "observation_id": "obs-1", "local_date": "2026-09-22"},
            "feedback_record_date_mismatch",
        ),
    ],
)
async def test_feedback_observation_missing_owner_or_date_fails(row, reason) -> None:
    svc, store = service()
    store.load_feedback_observation.return_value = row
    if reason == "feedback_record_not_found":
        result = await svc.get_plan_inputs(
            "u", "r", target_local_date="2026-09-23", feedback_record_id="obs-1"
        )
        assert result["data"]["feedback"]["status"] == "not_found"
        return
    with pytest.raises(DataError, match=reason):
        await svc.get_plan_inputs(
            "u", "r", target_local_date="2026-09-23", feedback_record_id="obs-1"
        )


async def test_feedback_observation_requires_target_date_but_not_service_consent() -> None:
    svc, store = service()
    with pytest.raises(DataError, match="feedback_target_date_required"):
        await svc.get_plan_inputs("u", "r", feedback_record_id="obs-1")
    store.load_feedback_observation.return_value = {
        "uid": "u",
        "observation_id": "obs-1",
        "local_date": "2026-09-23",
        "target_dates": ["2026-09-23"],
    }
    store.load_feedback_observation.return_value = None
    result = await svc.get_plan_inputs(
        "u", "r", target_local_date="2026-09-23", feedback_record_id="obs-1"
    )
    assert result["data"]["feedback"]["status"] == "not_found"
    store.load_feedback_observation.assert_awaited_once()
    store.load_permissions.assert_not_called()


async def test_target_plan_rejects_ambiguous_active_rows() -> None:
    svc, store = service()
    store.load_active_plan.return_value = [{"uid": "u"}, {"uid": "u"}]
    with pytest.raises(DataError, match="active_plan_ambiguous"):
        await svc.get_plan_inputs(
            "u", "request", inline_profile=profile(), target_local_date="2026-09-23"
        )


async def test_explicit_id_missing_is_not_hidden_by_service_consent() -> None:
    svc, store = service()
    store.load_profile.return_value = None, None
    with pytest.raises(DataError, match="profile_not_found"):
        await svc.get_plan_inputs("u", "req", profile_snapshot_id="missing")
    store.load_profile.assert_awaited_once()
    store.load_permissions.assert_not_awaited()
    store.reset_mock()
    store.load_permissions.return_value = {"status": "active", "privacy_switches": {}}
    with pytest.raises(DataError, match="profile_not_found"):
        await svc.get_plan_inputs("u", "req", profile_snapshot_id="missing")
    store.load_profile.assert_awaited_once()
    store.load_permissions.assert_not_called()


def test_cursor_signature_binding_and_expiration() -> None:
    svc, _store = service()
    binding = {"uid": "u", "episode_id": "e", "source": "s", "session_id": ""}
    token = svc._cursor_encode({**binding, "read_at": timestamp(utc_now()), "last": ["p", "s"]})
    assert svc._cursor_decode(token, binding)["last"] == ["p", "s"]
    for bad_token, bad_binding in (
        (token[:-1] + "x", binding),
        (token, {**binding, "uid": "other"}),
    ):
        with pytest.raises(DataError, match="invalid_cursor"):
            svc._cursor_decode(bad_token, bad_binding)
    old = svc._cursor_encode({**binding, "read_at": "2020-01-01T00:00:00Z", "last": ["p", "s"]})
    with pytest.raises(DataError, match="invalid_cursor"):
        svc._cursor_decode(old, binding)


async def test_exact_original_final_diff_and_revert() -> None:
    svc, store = service()
    original = {
        "uid": "u",
        "plan_id": "p",
        "version": 1,
        "snapshot_id": "v1",
        "content_hash": "h1",
        "phases": [{"phase": "Soothe", "nodes": [{"node_key": "main", "sound": {"volume": 30}}]}],
    }
    final = deepcopy(original)
    final.update(version=3, snapshot_id="v3", base_plan_version=1, edit_source="manual")
    final["phases"][0]["nodes"][0]["sound"]["volume"] = 15
    store.load_plan_revision.side_effect = lambda uid, plan, version, when: (
        original if version == 1 else final
    )
    store.list_execution_events.return_value = {
        "operations": [],
        "downlinks": [],
        "events": [],
        "session_payloads": [],
    }
    generation = {"output_plans": [{"plan_id": "p", "version": 1, "snapshot_id": "v1"}]}
    session = {
        "uid": "u",
        "session_id": "s",
        "episode_id": "e",
        "plan_id": "p",
        "plan_version": 3,
        "ended_at": "2026-09-20T00:00:00Z",
        "evidence_complete": True,
    }
    result = await svc._episode_diff(
        "u", session, datetime(2026, 9, 21, tzinfo=UTC), {}, generation
    )
    assert result["user_adjusted"] is True and result["user_adjustment_count"] == 1
    assert result["diff"][0]["before"] == 30 and result["diff"][0]["after"] == 15
    assert result["diff"][0]["origins"] == ["manual"]
    final["phases"] = deepcopy(original["phases"])
    result = await svc._episode_diff(
        "u", session, datetime(2026, 9, 21, tzinfo=UTC), {}, generation
    )
    assert result["user_adjusted"] is True and result["diff"] == []


async def test_weekly_requires_generation_before_queries() -> None:
    svc, store = service()
    with pytest.raises(DataError, match="weekly_generation_required"):
        await svc.get_profile_inputs(
            "u",
            "req",
            purpose="weekly",
            review_end_local_date="2026-09-21",
            timezone_offset="+08:00",
        )
    store.load_permissions.assert_not_called()


async def test_profile_refresh_does_not_apply_service_consent_gates() -> None:
    svc, store = service()
    svc.build_plan_comparisons = AsyncMock(return_value={"status": "ok", "value": []})
    svc._execution = AsyncMock(return_value={"status": "ok", "value": []})
    result = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="weekly",
        review_generation_id="g",
        review_end_local_date="2026-09-21",
        timezone_offset="+08:00",
        requested_sections=["plan_comparisons", "execution"],
    )
    assert result["data"]["plan_comparisons"]["status"] == "ok"
    assert result["data"]["execution"]["status"] == "ok"
    svc.build_plan_comparisons.assert_awaited_once()
    svc._execution.assert_awaited_once()
    store.load_permissions.assert_not_awaited()


async def test_execution_does_not_require_health_or_plan_consent() -> None:
    svc, store = service()
    store.load_permissions.return_value = {"status": "active", "privacy_switches": {"plans": True}}
    store.list_window_sessions.return_value = []
    store.list_execution_events.return_value = {}
    result = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="weekly",
        review_generation_id="g",
        review_end_local_date="2026-09-21",
        timezone_offset="+08:00",
        requested_sections=["execution"],
    )
    assert result["data"]["execution"]["status"] != "unsupported"
    store.list_window_sessions.assert_awaited_once()
    store.load_permissions.assert_not_called()


async def test_pagination_keeps_read_marker_without_permissions() -> None:
    svc, store = service()
    store.find_episode.return_value = {"episode_id": "e", "source": "s"}
    store.list_sessions.return_value = [
        {"uid": "u", "session_id": str(index), "plan_id": "p", "episode_id": "e"}
        for index in range(3)
    ]
    first = await svc.get_episode_evidence("u", "e", "s", limit=1)
    second = await svc.get_episode_evidence("u", "e", "s", limit=1, cursor=first["next_cursor"])
    assert first["plan_diffs"]["value"][0]["session_id"] == "0"
    assert second["plan_diffs"]["value"][0]["session_id"] == "1"
    assert first["read_started_at_utc"] == second["read_started_at_utc"]
    store.load_permissions.assert_not_awaited()
    store.load_permissions.assert_not_called()
    store.load_permissions.return_value = {"status": "active", "privacy_switches": {}}
    third = await svc.get_episode_evidence("u", "e", "s", cursor=second["next_cursor"])
    assert third["plan_diffs"]["value"][0]["session_id"] == "2"
    assert third["read_started_at_utc"] == first["read_started_at_utc"]
    store.load_permissions.assert_not_awaited()


async def test_all_profile_sections_share_actual_read_start_ignoring_legacy_cutoff() -> None:
    svc, store = service()
    cutoff = datetime(2026, 9, 20, tzinfo=UTC)
    methods = {
        "_questionnaires": 1,
        "_profile": 2,
        "build_facts": 2,
        "_observations": 2,
        "build_plan_comparisons": 2,
        "_execution": 2,
        "build_sleep_debt": 2,
    }
    for name in methods:
        setattr(svc, name, AsyncMock(return_value={"status": "not_found", "missing_reasons": []}))
    result = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="profile_refresh",
        review_end_local_date="2026-09-21",
        timezone_offset="+08:00",
        data_cutoff_at_utc=timestamp(cutoff),
    )
    for name, position in methods.items():
        assert (
            timestamp(getattr(svc, name).call_args.args[position]) == result["read_started_at_utc"]
        )
    assert result["read_started_at_utc"] != timestamp(cutoff)
    store.load_permissions.assert_not_called()


async def test_temporal_uncertainty_is_not_confirmed_cold_start() -> None:
    svc, _store = service()
    svc._profile = AsyncMock(side_effect=DataError("FAILED_PRECONDITION", "source_time_unverified"))
    result = await svc.get_profile_inputs(
        "u",
        "r",
        purpose="initial",
        review_end_local_date="2026-09-21",
        timezone_offset="+08:00",
        requested_sections=["prior_profile"],
    )
    assert result["data"]["prior_profile"]["status"] == "partial"
    assert result["data"]["prior_profile"]["missing_reasons"] == ["source_time_unverified"]


async def test_questionnaire_without_survey_version_preserves_actual_answer_fields() -> None:
    svc, store = service()
    store.load_questionnaires.return_value = [
        {
            "_id": "a",
            "answer_id": "a",
            "survey_code": "initial",
            "answers": [
                {
                    "question_id": "a0f15f100000000000000001",
                    "title": "年龄段",
                    "input_type": "radio",
                    "value": "synthetic-age",
                    "extra_input": "",
                }
            ],
            "received_at": datetime(2026, 10, 10, tzinfo=UTC),
        }
    ]
    result = await svc._questionnaires("u", datetime(2026, 10, 10, tzinfo=UTC), {})
    assert result["status"] == "ok"
    assert result["value"][0]["answers"][0] == {
        "question_id": "a0f15f100000000000000001",
        "title": "年龄段",
        "input_type": "radio",
        "value": "synthetic-age",
        "extra_input": "",
    }
    assert "survey_version" not in result["value"][0]
    assert result["missing_reasons"] == []


async def test_questionnaire_answers_include_selected_option_text() -> None:
    svc, store = service()
    store.load_quiz_questions.return_value = {
        "a0f15f100000000000000001": {
            "options": {
                "items": [
                    {"value": "A", "label": "18—24岁"},
                    {"value": "B", "label": "25—34岁"},
                ]
            }
        },
        "a0f15f100000000000000010": {
            "options": {
                "items": [
                    {"value": "B", "label": "越想越睡不着"},
                    {"value": "D", "label": "半夜容易醒"},
                ]
            }
        },
    }
    store.load_questionnaires.return_value = [
        {
            "_id": "a",
            "answer_id": "a",
            "survey_code": "initial",
            "answers": [
                {
                    "question_id": "a0f15f100000000000000001",
                    "title": "年龄段",
                    "input_type": "radio",
                    "value": "A",
                    "extra_input": "",
                },
                {
                    "question_id": "a0f15f100000000000000010",
                    "title": "睡眠困扰",
                    "input_type": "checkbox",
                    "value": ["B", "D"],
                    "extra_input": "",
                },
                {
                    "question_id": "legacy",
                    "title": "称呼",
                    "input_type": "radio",
                    "value": {"option_id": "A", "option_text": "先生"},
                },
                {
                    "question_id": "time",
                    "title": "作息",
                    "input_type": "time_picker",
                    "value": [{"value": "23:30", "label": "上床时间"}],
                },
            ],
            "received_at": datetime(2026, 10, 10, tzinfo=UTC),
        }
    ]
    result = await svc._questionnaires("u", datetime(2026, 10, 10, tzinfo=UTC), {})
    answers = result["value"][0]["answers"]
    assert answers[0]["value"] == "18—24岁"
    assert answers[1]["value"] == ["越想越睡不着", "半夜容易醒"]
    assert answers[2]["value"] == "先生"
    assert answers[3]["value"] == [{"value": "23:30", "label": "上床时间"}]


def test_current_profile_hashes_are_returned_without_comparison() -> None:
    import hashlib
    import json

    from app.server.somni.facts.models import validate_long_term_profile, validate_profile

    value = profile()
    value["schema_version"] = "0.0.1"
    value["long_term_profile"] = {"number": 30.0}
    encoded = json.dumps(value["long_term_profile"], sort_keys=True, separators=(",", ":"))
    value["long_term_content_hash"] = "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
    value.pop("content_hash")
    value["content_hash"] = canonical_hash(value)
    assert validate_profile(value, "u")["long_term_content_hash"] == value["long_term_content_hash"]
    value["long_term_profile"]["number"] = 31.0
    value["content_hash"] = "test-placeholder"
    validate_long_term_profile(value)
    accepted = validate_profile(value, "u")
    assert accepted["long_term_content_hash"] == value["long_term_content_hash"]
    assert accepted["content_hash"] == "test-placeholder"


async def test_prior_version_live_change_cannot_override_later_saved_version() -> None:
    svc, store = service()
    original = {
        "uid": "u",
        "plan_id": "p",
        "version": 1,
        "snapshot_id": "v1",
        "phases": [{"phase": "Soothe", "nodes": [{"node_key": "main", "sound": {"volume": 30}}]}],
    }
    final = deepcopy(original)
    final.update(version=2, snapshot_id="v2", base_plan_version=1, edit_source="manual")
    final["phases"][0]["nodes"][0]["sound"]["volume"] = 15
    store.load_plan_revision.side_effect = lambda uid, plan, version, when: (
        original if version == 1 else final
    )
    operation = {
        "op_id": "op1",
        "session_id": "s",
        "plan_id": "p",
        "plan_version": 1,
        "executed_at": "2026-09-20T01:00:00Z",
        "phase": "Soothe",
        "node_key": "main",
        "applied": True,
        "success": True,
        "origin": "manual",
        "parameters": {"sound": {"volume": 20}},
    }
    store.list_execution_events.return_value = {"operations": [operation]}
    session = {
        "uid": "u",
        "session_id": "s",
        "plan_id": "p",
        "plan_version": 1,
        "adopted_versions": [{"version": 2, "applied": True, "applied_at": "2026-09-20T02:00:00Z"}],
    }
    generation = {"output_plans": [{"plan_id": "p", "version": 1, "snapshot_id": "v1"}]}
    result = await svc._episode_diff(
        "u", session, datetime(2026, 9, 21, tzinfo=UTC), {}, generation
    )
    assert result["final_plan_ref"]["version"] == 2
    assert result["diff"][0]["after"] == 15
    assert result["execution_diff"] == []
