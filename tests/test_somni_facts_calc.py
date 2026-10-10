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
    select_last_effective_setting,
    summarize_execution,
)
from app.server.somni.facts.time import resolve_scope


def test_numeric_missing_zero_and_mad() -> None:
    assert aggregate_numeric([390, 420, 450])["mad"] == 30
    assert aggregate_numeric([390, 420, 450])["mean"] == 420
    assert aggregate_numeric([None, float("nan"), float("inf"), True])["mean"] is None
    assert aggregate_numeric([0])["mad"] is None
    assert aggregate_numeric([0, 0])["mad"] == 0


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
