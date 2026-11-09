from copy import deepcopy
from datetime import UTC, datetime

import pytest

from app.server.somni.facts.calc import (
    aggregate_clock,
    aggregate_environment,
    aggregate_numeric,
    build_series,
    calculate_laeq,
    compare_plan_execution,
    diff_plan_versions,
    normalize_daily_metrics,
    normalize_metric_windows,
    normalize_somni_record_metrics,
    select_last_effective_setting,
    select_somni_record_rows,
    summarize_execution,
)
from app.server.somni.facts.time import resolve_scope


def test_numeric_missing_zero_and_mad() -> None:
    assert aggregate_numeric([390, 420, 450])["mad"] == 30
    assert aggregate_numeric([390, 420, 450])["mean"] == 420
    assert aggregate_numeric([None, float("nan"), float("inf"), True])["mean"] is None
    assert aggregate_numeric([0])["mad"] is None
    assert aggregate_numeric([0, 0])["mad"] == 0


def test_presleep_windows_require_full_coverage_and_provenance() -> None:
    window = {
        "metric_id": "presleep_hr_median_bpm",
        "unit": "bpm",
        "source": "radar",
        "method": "median",
        "definition_version": "v6",
        "local_date": "2026-10-12",
        "episode_id": "test-episode",
        "start_at_utc": "2026-10-12T14:00:00Z",
        "end_at_utc": "2026-10-12T14:01:00Z",
        "value": 60,
        "quality": "valid",
        "covered_sec": 60,
        "sample_count": 12,
        "measurement_condition": "presleep_awake_rest",
    }
    row = {"record_date": "2026-10-12", "source_refs": ["test:source"], "metric_windows": [window]}
    cutoff = datetime(2026, 10, 13, tzinfo=UTC)
    values, reasons = normalize_metric_windows([row], cutoff)
    assert len(values) == 1 and not reasons
    invalid = {**row, "metric_windows": [{**window, "covered_sec": 59}]}
    assert not normalize_metric_windows([invalid], cutoff)[0]
    future = datetime(2026, 10, 12, 13, tzinfo=UTC)
    assert not normalize_metric_windows([row], future)[0]
    conflict = {**row, "metric_windows": [window, {**window, "value": 61}]}
    values, reasons = normalize_metric_windows([conflict], cutoff)
    assert not values and "metric_window_conflict" in reasons


def test_somni_records_project_report_compatible_sleep_metrics() -> None:
    row = {
        "_id": "record-1",
        "source_type": "records",
        "record_date": "2026-10-07",
        "main_sleep": True,
        "timezone_offset": "+08:00",
        "source_refs": ["records:record-1"],
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

    metrics, reasons = normalize_somni_record_metrics(
        [row], timezone_offset="+08:00", read_at=datetime(2026, 10, 8, tzinfo=UTC)
    )

    assert not reasons
    assert {item["metric_id"]: item["value"] for item in metrics} == {
        "total_sleep_min": 456,
        "sol_min": 20,
        "waso_min": 120,
        "sleep_efficiency_pct": 85,
    }
    assert all(item["local_date"] == "2026-10-07" for item in metrics)
    assert all(item["source_refs"] == ["records:record-1"] for item in metrics)

    unverified_main_sleep = deepcopy(row)
    del unverified_main_sleep["main_sleep"]
    metrics, reasons = normalize_somni_record_metrics(
        [unverified_main_sleep],
        timezone_offset="+08:00",
        read_at=datetime(2026, 10, 8, tzinfo=UTC),
    )
    assert not metrics
    assert "somni_record_main_sleep_unverified" in reasons

    missing_latency = deepcopy(row)
    del missing_latency["raw_data"]["sleep_latency"]
    metrics, reasons = normalize_somni_record_metrics(
        [missing_latency],
        timezone_offset="+08:00",
        read_at=datetime(2026, 10, 8, tzinfo=UTC),
    )
    assert "sol_min" not in {item["metric_id"] for item in metrics}
    assert "somni_record_sleep_latency_unverified" in reasons

    explicit_zero_latency = deepcopy(row)
    explicit_zero_latency["raw_data"]["sleep_latency"] = 0
    metrics, _ = normalize_somni_record_metrics(
        [explicit_zero_latency],
        timezone_offset="+08:00",
        read_at=datetime(2026, 10, 8, tzinfo=UTC),
    )
    assert next(item["value"] for item in metrics if item["metric_id"] == "sol_min") == 0

    invalid_quality = deepcopy(row)
    invalid_quality["quality"] = {"valid": False}
    metrics, reasons = normalize_somni_record_metrics(
        [invalid_quality],
        timezone_offset="+08:00",
        read_at=datetime(2026, 10, 8, tzinfo=UTC),
    )
    assert not metrics
    assert "somni_record_quality_unverified" in reasons

    after_cutoff = deepcopy(row)
    after_cutoff["raw_data"]["wake_up_time"] = datetime(2026, 10, 8, 1, tzinfo=UTC)
    metrics, reasons = normalize_somni_record_metrics(
        [after_cutoff],
        timezone_offset="+08:00",
        read_at=datetime(2026, 10, 8, tzinfo=UTC),
    )
    assert not metrics
    assert "somni_record_sleep_interval_unverified" in reasons

    mismatched = {**row, "record_date": "2026-10-06"}
    metrics, reasons = normalize_somni_record_metrics(
        [mismatched], timezone_offset="+08:00", read_at=datetime(2026, 10, 8, tzinfo=UTC)
    )
    assert not metrics
    assert "somni_record_wake_date_mismatch" in reasons


def test_select_somni_record_rows_requires_unique_episode_revision() -> None:
    earlier = {
        "record_date": "2026-10-07",
        "source": "radar",
        "episode_id": "episode-1",
        "revision": 1,
        "main_sleep": True,
    }
    latest = {**earlier, "revision": 2}
    selected, reasons = select_somni_record_rows([earlier, latest])
    assert selected == [latest] and not reasons

    ambiguous = {**earlier, "episode_id": "episode-2", "revision": 1}
    selected, reasons = select_somni_record_rows([earlier, ambiguous])
    assert not selected
    assert reasons == ["somni_record_revision_ambiguous"]

    nap = {**earlier, "main_sleep": False}
    selected, reasons = select_somni_record_rows([nap])
    assert not selected
    assert reasons == ["somni_record_not_main_sleep"]


def test_clocks_and_laeq() -> None:
    assert aggregate_clock([1430, 10])["circular_mean_min"] == pytest.approx(0)
    assert aggregate_clock([0, 720])["circular_mean_min"] is None
    assert aggregate_clock([0])["circular_mean_abs_deviation_min"] is None
    assert calculate_laeq([30, 40], [1, 1]) == pytest.approx(37.40362689)
    assert calculate_laeq([], []) is None


def test_time_scope() -> None:
    scope = resolve_scope("2026-09-21", "+08:00")
    assert scope.start(7).isoformat() == "2026-09-14"
    assert scope.start(28).isoformat() == "2026-08-24"
    assert scope.bounds(7)[1].isoformat() == "2026-09-20T16:00:00+00:00"


@pytest.mark.parametrize("offset", ["+14:01", "-12:01", "+08:60", "08:00"])
def test_invalid_offset(offset: str) -> None:
    with pytest.raises(ValueError):
        resolve_scope("2026-09-21", offset)


def test_environment_deduplicates_intervals() -> None:
    windows = [{"start": 0, "end": 10, "value": 20}] * 2
    assert aggregate_environment(windows, 0, 20)["covered_sec"] == 10
    assert aggregate_environment(windows, 0, 20)["mean"] == 20
    with pytest.raises(ValueError):
        aggregate_environment(windows + [{"start": 0, "end": 5, "value": 30}], 0, 20)


def test_diff_stable_keys_missing_null_and_restore() -> None:
    original = {
        "version": 1,
        "phases": [
            {
                "phase": "Soothe",
                "nodes": [{"node_key": "main/a", "sound": {"volume": 30, "muted": False}}],
            }
        ],
    }
    final = deepcopy(original)
    final["version"] = 3
    node = final["phases"][0]["nodes"][0]
    node["sound"]["volume"] = 15
    node["sound"]["nullable"] = None
    changes = diff_plan_versions(original, final)
    assert any(
        row["path"].endswith("main~1a/sound/volume") and row["after"] == 15 for row in changes
    )
    added = next(row for row in changes if row["op"] == "add")
    assert not added["before_present"] and added["after_present"] and added["after"] is None
    assert diff_plan_versions(original, {**original, "version": 9}) == []


def test_series_missing_quality_and_hrv_isolation() -> None:
    rows = [
        {
            "source": "synthetic",
            "record_date": "2026-09-20",
            "device_algorithm": "v1",
            "definition_version": "1",
            "main_sleep": True,
            "quality": {"valid": True},
            "sleep": {"total_asleep_min": 420},
            "physiology": {"hrv_sleep_avg_ms": 30},
        }
    ]
    metrics = normalize_daily_metrics(rows)
    hrv = next(item for item in metrics if item["metric_id"] == "sleep_hrv_mean_ms")
    assert "value" not in hrv and hrv["missing_reason"] == "hrv_method_unknown"
    series = build_series(metrics, resolve_scope("2026-09-21", "+08:00"))
    sleep = next(item for item in series if item["metric_id"] == "total_sleep_min")
    assert sleep["stats_7d"]["valid_count"] == 1
    assert sleep["stats_7d"]["missing_count"] == 6
    assert sleep["stats_7d"]["numeric"]["mean"] == 420
    assert "mad" not in sleep["stats_7d"]["numeric"]


def test_build_series_combines_fallback_sources_by_date() -> None:
    scope = resolve_scope("2026-10-08", "+00:00")
    metrics = [
        {
            "metric_id": "total_sleep_min",
            "unit": "min",
            "source": source,
            "method": method,
            "definition_version": "v1",
            "local_date": day,
            "quality": "valid",
            "value": value,
            "source_refs": [f"{source}:{day}"],
        }
        for source, method, day, value in (
            ("somni_records", "record_projection_v1", "2026-10-01", 390),
            ("somni_records", "record_projection_v1", "2026-10-02", 420),
            ("sleep_reports", "daily_report_projection_v1", "2026-10-03", 450),
        )
    ]

    series = build_series(metrics, scope)

    assert len(series) == 1
    assert series[0]["stats_7d"]["valid_count"] == 3
    assert series[0]["stats_7d"]["missing_count"] == 4
    assert series[0]["stats_7d"]["numeric"]["mean"] == 420
    assert [item["source"] for item in series[0]["daily"][:3]] == [
        "somni_records",
        "somni_records",
        "sleep_reports",
    ]


def test_build_series_preserves_refs_for_ambiguous_daily_values() -> None:
    scope = resolve_scope("2026-10-08", "+00:00")
    metrics = [
        {
            "metric_id": "total_sleep_min",
            "unit": "min",
            "source": source,
            "method": "daily_v1",
            "definition_version": "v1",
            "local_date": "2026-10-07",
            "quality": "valid",
            "value": value,
            "source_refs": [f"{source}:2026-10-07"],
        }
        for source, value in (("somni_records", 420), ("somni_sleep_reports", 400))
    ]

    series = build_series(metrics, scope)

    daily = series[0]["daily"][0]
    assert daily["quality"] == "missing"
    assert daily["missing_reason"] == "ambiguous_daily_value"
    assert "value" not in daily
    assert set(daily["source_refs"]) == {
        "somni_records:2026-10-07",
        "somni_sleep_reports:2026-10-07",
    }


def test_confirmed_changes_exclude_failure_and_terminal_zero() -> None:
    final = {
        "phases": [{"phase": "Soothe", "nodes": [{"node_key": "main", "sound": {"volume": 30}}]}]
    }

    def operation(index, volume, **changes):
        return {
            "op_id": str(index),
            "executed_at": f"2026-09-20T0{index}:00:00Z",
            "applied": True,
            "success": True,
            "origin": "manual",
            "phase": "Soothe",
            "node_key": "main",
            "parameters": {"sound": {"volume": volume}},
            **changes,
        }

    operations = [
        operation(1, 20),
        operation(2, 15, success=False),
        operation(3, 0, origin="automation", reason="normal_end"),
    ]
    read_at = datetime(2026, 9, 21, tzinfo=UTC)
    assert select_last_effective_setting(operations, read_at)["op_id"] == "1"
    assert compare_plan_execution(final, operations, read_at)[0]["after"] == 20
    operations.append(operation(4, 0))
    assert compare_plan_execution(final, operations, read_at)[0]["after"] == 0


def test_execution_intervals_do_not_count_delivery_or_failed_commands() -> None:
    session = {"session_id": "s", "plan_id": "p", "ended_at": "2026-09-20T03:00:00Z"}
    common = {"session_id": "s", "phase": "Soothe", "node_index": 0, "plan_version": 1}
    rows = [
        {
            **common,
            "op_id": "1",
            "executed_at": "2026-09-20T01:00:00Z",
            "ack": True,
            "success": True,
            "applied": True,
            "parameters": {"enabled": True},
        },
        {
            **common,
            "op_id": "2",
            "executed_at": "2026-09-20T02:00:00Z",
            "ack": True,
            "success": False,
            "applied": False,
            "parameters": {"enabled": False},
        },
    ]
    items = summarize_execution(session, rows + rows, datetime(2026, 9, 21, tzinfo=UTC))
    assert items[0]["command_count"] == 2
    assert items[0]["confirmed_on_sec"] == 7200
    assert "measured_on_sec" not in items[0]


def test_diff_numeric_equivalence_and_ambiguous_alignment() -> None:
    assert (
        diff_plan_versions({"phases": [{"duration": 30}]}, {"phases": [{"duration": 30.0}]}) == []
    )
    change = diff_plan_versions({"phases": [{"duration": 30}]}, {"phases": [{"duration": 15}]})[0]
    assert change["missing_reasons"] == ["subtree_replaced_without_stable_alignment"]


def test_execution_origin_is_per_parameter_not_last_modality() -> None:
    final = {
        "phases": [
            {"phase": "Soothe", "nodes": [{"node_key": "main", "sound": {"volume": 30, "pan": 0}}]}
        ]
    }
    common = {"applied": True, "success": True, "phase": "Soothe", "node_key": "main"}
    operations = [
        {
            **common,
            "op_id": "a",
            "origin": "manual",
            "executed_at": "2026-09-20T01:00:00Z",
            "parameters": {"sound": {"volume": 15}},
        },
        {
            **common,
            "op_id": "b",
            "origin": "automation",
            "executed_at": "2026-09-20T02:00:00Z",
            "parameters": {"sound": {"pan": 1}},
        },
    ]
    changes = compare_plan_execution(final, operations, datetime(2026, 9, 21, tzinfo=UTC))
    assert next(row for row in changes if row["path"].endswith("/volume"))["origins"] == ["manual"]
    assert next(row for row in changes if row["path"].endswith("/pan"))["origins"] == ["automation"]


def test_conflicting_simultaneous_application_does_not_double_duration() -> None:
    session = {"session_id": "s", "ended_at": "2026-09-20T03:00:00Z"}
    common = {
        "session_id": "s",
        "executed_at": "2026-09-20T01:00:00Z",
        "phase": "Soothe",
        "node_index": 0,
        "applied": True,
        "success": True,
        "parameters": {"enabled": True},
    }
    result = summarize_execution(
        session,
        [{**common, "op_id": "a"}, {**common, "op_id": "b"}],
        datetime(2026, 9, 21, tzinfo=UTC),
    )
    assert "confirmed_on_sec" not in result[0]
