from __future__ import annotations

import math
from copy import deepcopy
from datetime import timedelta
from statistics import mean, median
from typing import Any

from app.server.somni.facts.time import (
    TimeScope,
    attribute_sleep_date,
    instant,
    parse_date,
    visible_by,
)
from app.server.somni.report.calc import minutes_between, sleep_stage_parts


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate_numeric(values: list[float | None]) -> dict[str, float | None]:
    valid = [value for value in values if finite_number(value)]
    if not valid:
        return dict.fromkeys(("mean", "median", "mad", "min", "max"))
    center = median(valid)
    return {
        "mean": mean(valid),
        "median": center,
        "mad": median(abs(value - center) for value in valid) if len(valid) > 1 else None,
        "min": min(valid),
        "max": max(valid),
    }


def normalize_metric_windows(rows: list[dict], read_at: Any) -> tuple[list[dict], list[str]]:
    """Expose only explicitly sourced, fully covered physiological windows."""
    units = {
        "presleep_hr_median_bpm": "bpm",
        "presleep_resp_median_per_min": "breaths/min",
        "presleep_hrv_rmssd_ms": "ms",
    }
    selected, conflicts, reasons = {}, set(), set()
    fields = (
        "metric_id unit source method definition_version local_date episode_id "
        "start_at_utc end_at_utc value quality covered_sec sample_count "
        "measurement_condition hrv_method source_refs"
    ).split()
    for row in rows:
        raw = row.get("raw_data")
        windows = row.get(
            "metric_windows", raw.get("metric_windows", []) if isinstance(raw, dict) else []
        )
        if not isinstance(windows, list):
            reasons.add("metric_windows_invalid")
            continue
        for window in windows:
            if not isinstance(window, dict):
                reasons.add("metric_window_invalid")
                continue
            item = {key: window[key] for key in fields if key in window}
            start, end = instant(item.get("start_at_utc")), instant(item.get("end_at_utc"))
            count = item.get("sample_count")
            duration = (end - start).total_seconds() if start and end else 0
            if (
                item.get("metric_id") not in units
                or item.get("unit") != units.get(item.get("metric_id"))
                or item.get("quality") != "valid"
                or item.get("measurement_condition") != "presleep_awake_rest"
                or not all(
                    item.get(key)
                    for key in ("source", "method", "definition_version", "episode_id")
                )
                or item.get("local_date") != row.get("record_date")
                or not start
                or not end
                or duration <= 0
                or not visible_by(end, read_at)
                or not finite_number(item.get("value"))
                or item["value"] <= 0
                or not finite_number(item.get("covered_sec"))
                or item["covered_sec"] != duration
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
                or (
                    item["metric_id"] == "presleep_hrv_rmssd_ms"
                    and item.get("hrv_method") != "RMSSD"
                )
            ):
                reasons.add("metric_window_quality_or_provenance_unverified")
                continue
            refs = row.get("source_refs") or item.get("source_refs")
            if (
                not isinstance(refs, list)
                or not refs
                or not all(isinstance(ref, str) and ref for ref in refs)
            ):
                reasons.add("metric_window_source_refs_missing")
                continue
            item.update(
                start_at_utc=start.isoformat(), end_at_utc=end.isoformat(), source_refs=refs
            )
            key = tuple(
                item.get(field)
                for field in (
                    "metric_id",
                    "local_date",
                    "episode_id",
                    "source",
                    "method",
                    "definition_version",
                    "start_at_utc",
                    "end_at_utc",
                    "measurement_condition",
                    "hrv_method",
                )
            )
            if key in selected and selected[key] != item:
                conflicts.add(key)
                reasons.add("metric_window_conflict")
            else:
                selected[key] = item
    return [item for key, item in selected.items() if key not in conflicts], sorted(reasons)


def aggregate_clock(values: list[float | None]) -> dict[str, float | None]:
    valid = [value for value in values if finite_number(value) and 0 <= value < 1440]
    result = dict.fromkeys(
        ("circular_mean_min", "circular_mean_abs_deviation_min", "concentration")
    )
    if not valid:
        return result
    sine = mean(math.sin(value * math.tau / 1440) for value in valid)
    cosine = mean(math.cos(value * math.tau / 1440) for value in valid)
    concentration = math.hypot(sine, cosine)
    result["concentration"] = concentration
    if concentration < 1e-6:
        return result
    center = (math.atan2(sine, cosine) * 1440 / math.tau) % 1440
    if math.isclose(center, 1440):
        center = 0.0
    result["circular_mean_min"] = center
    if len(valid) > 1:
        result["circular_mean_abs_deviation_min"] = mean(
            abs((value - center + 720) % 1440 - 720) for value in valid
        )
    return result


def calculate_laeq(levels: list[float], durations: list[float]) -> float | None:
    if len(levels) != len(durations):
        raise ValueError("mismatched_intervals")
    pairs = [
        (level, duration)
        for level, duration in zip(levels, durations, strict=True)
        if finite_number(level) and finite_number(duration) and duration > 0
    ]
    if not pairs:
        return None
    maximum = max(level for level, _duration in pairs)
    energy = sum(duration * 10 ** ((level - maximum) / 10) for level, duration in pairs)
    return maximum + 10 * math.log10(energy / sum(duration for _level, duration in pairs))


def aggregate_environment(windows: list[dict], start: float, end: float) -> dict:
    valid = [
        row
        for row in windows
        if all(finite_number(row.get(key)) for key in ("start", "end", "value"))
        and row["end"] > row["start"]
    ]
    boundaries = sorted(
        {start, end} | {max(start, min(end, row[key])) for row in valid for key in ("start", "end")}
    )
    weighted = covered = 0.0
    observed = []
    for left, right in zip(boundaries, boundaries[1:]):
        active = [row for row in valid if row["start"] <= left and row["end"] >= right]
        if not active:
            continue
        values = {row["value"] for row in active}
        if len(values) > 1:
            raise ValueError("conflicting_signal_windows")
        value = active[0]["value"]
        weighted += value * (right - left)
        covered += right - left
        observed.append(value)
    return {
        "mean": weighted / covered if covered else None,
        "min": min(observed) if observed else None,
        "max": max(observed) if observed else None,
        "covered_sec": covered,
        "sample_count": len(valid),
    }


def _escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _normalize(value: Any, kind: str = "") -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize(item, key)
            for key, item in value.items()
            if key not in ("created_at", "updated_at", "editor_id", "content_hash", "lifecycle")
        }
    if isinstance(value, list):
        identity = {"phases": "phase", "nodes": "node_key", "tracks": "track_id"}.get(kind)
        if (
            identity
            and value
            and all(isinstance(item, dict) and item.get(identity) for item in value)
        ):
            keys = [str(item[identity]) for item in value]
            if len(set(keys)) == len(keys) and "order" not in keys:
                return {
                    "order": keys,
                    **{key: _normalize(item) for key, item in zip(keys, value, strict=True)},
                }
        return [_normalize(item) for item in value]
    return deepcopy(value)


def diff_plan_versions(original: dict, final: dict) -> list[dict]:
    if not isinstance(original.get("phases"), list) or not isinstance(final.get("phases"), list):
        raise ValueError("missing_plan_phases")
    executable = ("phases", "execution_time", "plan_start_local_date", "timezone_offset")
    before = _normalize({key: original[key] for key in executable if key in original})
    after = _normalize({key: final[key] for key in executable if key in final})
    changes = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                pointer = path + "/" + _escape(key)
                if key in left and key in right:
                    walk(left[key], right[key], pointer)
                else:
                    changes.append(
                        {
                            "op": "remove" if key in left else "add",
                            "path": pointer,
                            "before_present": key in left,
                            "after_present": key in right,
                            "before": left.get(key),
                            "after": right.get(key),
                            "origins": ["unknown"],
                        }
                    )
        elif (
            type(left) is not type(right) and not (finite_number(left) and finite_number(right))
        ) or left != right:
            changes.append(
                {
                    "op": "replace",
                    "path": path,
                    "before_present": True,
                    "after_present": True,
                    "before": left,
                    "after": right,
                    "origins": ["unknown"],
                }
            )

    walk(before, after, "")
    for change in changes:
        parts = [
            part.replace("~1", "/").replace("~0", "~") for part in change["path"].split("/")[1:]
        ]
        if len(parts) > 1 and parts[0] == "phases" and parts[1] != "order":
            change["phase"] = parts[1]
        if len(parts) > 3 and parts[2] == "nodes" and parts[3] != "order":
            change["node_key"] = parts[3]
        if len(parts) > 4 and parts[4] in ("sound", "light", "scent"):
            change["modality"] = parts[4]
        if (
            change["op"] == "replace"
            and (isinstance(change["before"], list) or isinstance(change["after"], list))
            and not change["path"].endswith("/order")
        ):
            change["missing_reasons"] = ["subtree_replaced_without_stable_alignment"]
    return changes


METRIC_PATHS = {
    "total_sleep_min": ("sleep.total_asleep_min", "min"),
    "waso_min": ("sleep.waso_min", "min"),
    "awakening_count": ("sleep.awakening_count", "count"),
    "deep_min": ("sleep.deep_min", "min"),
    "rem_min": ("sleep.rem_min", "min"),
    "core_min": ("sleep.core_min", "min"),
    "unspecified_asleep_min": ("sleep.unspecified_asleep_min", "min"),
    "sleep_hr_mean_bpm": ("physiology.heart_rate_sleep_avg_bpm", "bpm"),
    "sleep_hr_min_bpm": ("physiology.heart_rate_sleep_min_bpm", "bpm"),
    "sleep_hr_max_bpm": ("physiology.heart_rate_sleep_max_bpm", "bpm"),
    "sleep_hrv_mean_ms": ("physiology.hrv_sleep_avg_ms", "ms"),
    "sleep_resp_mean_per_min": ("physiology.respiratory_rate_sleep_avg", "per_min"),
    "day_steps": ("activity.windows.calendar_day.steps", "count"),
    "day_exercise_min": ("activity.windows.calendar_day.exercise_min", "min"),
}


def _get_path(row: dict, path: str) -> Any:
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def normalize_daily_metrics(rows: list[dict], *, activity: bool = False) -> list[dict]:
    result = []
    for row in rows:
        local_date = row.get("activity_date" if activity else "record_date")
        if not local_date or not row.get("source"):
            continue
        for metric, (path, unit) in METRIC_PATHS.items():
            if path.startswith("activity.") != activity:
                continue
            value = _get_path(row, path)
            reason = ""
            if not finite_number(value) or value < 0:
                reason = "missing_or_invalid_value"
            elif not row.get("definition_version") or not row.get("device_algorithm"):
                reason = "metric_definition_unknown"
            elif not activity and row.get("main_sleep") is not True:
                reason = "main_sleep_unverified"
            elif metric == "sleep_hrv_mean_ms" and row.get("hrv_method") not in ("SDNN", "RMSSD"):
                reason = "hrv_method_unknown"
            elif row.get("quality", {}).get("valid") is not True:
                reason = "quality_unverified"
            item = {
                "metric_id": metric,
                "unit": unit,
                "source": row["source"],
                "method": "source_daily",
                "definition_version": row.get("definition_version", ""),
                "device_algorithm": row.get("device_algorithm", ""),
                "hrv_method": row.get("hrv_method", "") if "hrv" in metric else "",
                "local_date": local_date,
                "quality": "missing" if reason else "valid",
                "source_refs": row.get("source_refs", []),
            }
            if row.get("episode_id"):
                item["episode_id"] = row["episode_id"]
            if reason:
                item["missing_reason"] = reason
            else:
                item["value"] = value
            result.append(item)
    return result


def select_somni_record_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Select one daily record or a uniquely revised version of one episode."""
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        local_date = row.get("record_date")
        try:
            local_date = parse_date(local_date).isoformat()
        except (TypeError, ValueError):
            continue
        by_date.setdefault(local_date, []).append(row)

    selected: list[dict] = []
    reasons: set[str] = set()
    for candidates in by_date.values():
        if len(candidates) == 1:
            candidate = candidates[0]
        else:
            identities = {(row.get("source"), row.get("episode_id")) for row in candidates}
            revisions = [row.get("revision") for row in candidates]
            if (
                len(identities) != 1
                or not all(isinstance(value, str) and value for value in next(iter(identities)))
                or not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 1
                    for value in revisions
                )
                or len(set(revisions)) != len(revisions)
            ):
                reasons.add("somni_record_revision_ambiguous")
                continue
            candidate = max(candidates, key=lambda row: row["revision"])
        if candidate.get("main_sleep") is False:
            reasons.add("somni_record_not_main_sleep")
            continue
        if candidate.get("main_sleep") is not True:
            reasons.add("somni_record_main_sleep_unverified")
            continue
        selected.append(candidate)
    return selected, sorted(reasons)


def normalize_somni_record_metrics(
    rows: list[dict], *, timezone_offset: str, read_at: Any
) -> tuple[list[dict], list[str]]:
    """Project validated legacy record fields into daily sleep facts."""
    result = []
    reasons: set[str] = set()
    cutoff = instant(read_at)
    if cutoff is None:
        return [], ["somni_record_cutoff_unverified"]

    for row in rows:
        if row.get("source_type") != "records":
            continue
        if row.get("main_sleep") is not True:
            reasons.add(
                "somni_record_not_main_sleep"
                if row.get("main_sleep") is False
                else "somni_record_main_sleep_unverified"
            )
            continue
        raw = row.get("raw_data")
        if not isinstance(raw, dict):
            reasons.add("somni_record_raw_data_missing")
            continue
        record_date = row.get("record_date")
        try:
            record_date = parse_date(record_date).isoformat()
        except (TypeError, ValueError):
            reasons.add("somni_record_date_invalid")
            continue
        quality = row.get("quality")
        if quality is not None and (
            not isinstance(quality, dict) or quality.get("valid") is not True
        ):
            reasons.add("somni_record_quality_unverified")
            continue

        bed = instant(raw.get("bed_time"))
        wake = instant(raw.get("wake_time"))
        wake_up = instant(raw.get("wake_up_time"))
        if not bed or not wake_up or bed >= wake_up or not visible_by(wake_up, cutoff):
            reasons.add("somni_record_sleep_interval_unverified")
            continue
        offset = str(row.get("timezone_offset") or timezone_offset)
        try:
            wake_date = attribute_sleep_date(wake_up, offset)
        except ValueError:
            reasons.add("somni_record_timezone_unverified")
            continue
        if wake_date != record_date:
            reasons.add("somni_record_wake_date_mismatch")
            continue
        bed_minutes = minutes_between(bed, wake_up)
        if bed_minutes <= 0:
            reasons.add("somni_record_sleep_interval_invalid")
            continue

        refs = row.get("source_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or not all(isinstance(ref, str) and ref for ref in refs)
        ):
            reasons.add("somni_record_source_refs_missing")
            continue

        source = str(row.get("source") or "somni_records")
        common: dict[str, Any] = {
            "unit": "min",
            "source": source,
            "local_date": record_date,
            "quality": "valid",
            "source_refs": refs,
        }
        if row.get("episode_id"):
            common["episode_id"] = str(row["episode_id"])

        ratio_keys = ("awake_ratio", "rem_ratio", "light_sleep_ratio", "deep_sleep_ratio")
        ratios = {key: raw.get(key) for key in ratio_keys}
        if all(finite_number(value) and 0 <= value <= 100 for value in ratios.values()):
            parts = sleep_stage_parts(raw)
            total_sleep = parts["total_minutes"]
            if 0 <= total_sleep <= bed_minutes:
                result.append(
                    {
                        **common,
                        "metric_id": "total_sleep_min",
                        "method": "somni_report_stage_projection_v1",
                        "definition_version": "somni_records_raw_data_v1",
                        "value": total_sleep,
                    }
                )
            else:
                reasons.add("somni_record_sleep_stage_duration_invalid")
        else:
            reasons.add("somni_record_sleep_stage_ratios_unverified")

        latency = raw.get("sleep_latency")
        if finite_number(latency) and 0 <= latency <= bed_minutes:
            result.append(
                {
                    **common,
                    "metric_id": "sol_min",
                    "method": "somni_record_sleep_latency_v1",
                    "definition_version": "somni_records_raw_data_v1",
                    "value": int(latency),
                }
            )
        else:
            reasons.add("somni_record_sleep_latency_unverified")

        efficiency = raw.get("sleep_efficiency")
        if finite_number(efficiency) and 0 <= efficiency <= 100:
            result.append(
                {
                    **common,
                    "unit": "pct",
                    "metric_id": "sleep_efficiency_pct",
                    "method": "somni_record_sleep_efficiency_v1",
                    "definition_version": "somni_records_raw_data_v1",
                    "value": int(efficiency),
                }
            )
        else:
            reasons.add("somni_record_sleep_efficiency_unverified")

        if wake and bed <= wake <= wake_up:
            result.append(
                {
                    **common,
                    "metric_id": "waso_min",
                    "method": "somni_record_awake_interval_v1",
                    "definition_version": "somni_records_raw_data_v1",
                    "value": minutes_between(wake, wake_up),
                }
            )
        else:
            reasons.add("somni_record_waso_interval_unverified")

    return result, sorted(reasons)


def build_series(metrics: list[dict], scope: TimeScope) -> list[dict]:
    grouped = {}
    for metric in metrics:
        if not scope.start(28).isoformat() <= metric["local_date"] < scope.end.isoformat():
            continue
        key = (metric.get("metric_id", ""), metric.get("unit", ""))
        grouped.setdefault(key, {}).setdefault(metric["local_date"], []).append(metric)
    result = []
    for key, days in sorted(grouped.items()):
        daily = []
        for local_date, candidates in sorted(days.items()):
            item = dict(candidates[0])
            if len(candidates) > 1:
                item.pop("value", None)
                item.update(quality="missing", missing_reason="ambiguous_daily_value")
                item["source_refs"] = list(
                    dict.fromkeys(
                        ref
                        for candidate in candidates
                        for ref in candidate.get("source_refs", [])
                        if isinstance(ref, str) and ref
                    )
                )
                for field in (
                    "source",
                    "method",
                    "device_algorithm",
                    "definition_version",
                    "hrv_method",
                ):
                    if len({str(candidate.get(field) or "") for candidate in candidates}) > 1:
                        item[field] = "mixed"
            daily.append(item)
        indexed = {item["local_date"]: item for item in daily}
        series = {"metric_id": key[0], "unit": key[1], "daily": daily}
        source_metrics = [item for candidates in days.values() for item in candidates]
        for field in ("source", "device_algorithm", "method", "definition_version", "hrv_method"):
            values = {str(item.get(field) or "") for item in source_metrics}
            if len(values) == 1:
                value = next(iter(values))
                if value:
                    series[field] = value
            elif values:
                series[field] = "mixed"
        for window in (7, 28):
            values, missing = [], {}
            for index in range(window):
                local_date = (scope.start(window) + timedelta(days=index)).isoformat()
                item = indexed.get(local_date, {})
                if finite_number(item.get("value")):
                    values.append(item["value"])
                else:
                    reason = item.get("missing_reason", "no_daily_record")
                    missing[reason] = missing.get(reason, 0) + 1
            clock = key[0].endswith("_clock")
            stats = aggregate_clock(values) if clock else aggregate_numeric(values)
            series[f"stats_{window}d"] = {
                "start_local_date": scope.start(window).isoformat(),
                "end_local_date_exclusive": scope.end.isoformat(),
                "window_days": window,
                "valid_count": len(values),
                "missing_count": window - len(values),
                "missing_reason_counts": missing,
                "clock" if clock else "numeric": {
                    name: value for name, value in stats.items() if value is not None
                },
            }
        valid_dates = [item["local_date"] for item in daily if finite_number(item.get("value"))]
        coverage = {
            "covered_span_days": 0,
            "recent_7d_valid_count": series["stats_7d"]["valid_count"],
        }
        if valid_dates:
            coverage.update(
                first_valid_local_date=valid_dates[0],
                last_valid_local_date=valid_dates[-1],
                covered_span_days=(parse_date(valid_dates[-1]) - parse_date(valid_dates[0])).days
                + 1,
            )
        series["coverage_28d"] = coverage
        result.append(series)
    return result


def _applied(operation: dict) -> bool:
    return operation.get("applied") is True and operation.get("success") is True


def select_last_effective_setting(operations: list[dict], read_at: Any = None) -> dict | None:
    valid = [
        row
        for row in operations
        if _applied(row)
        and instant(row.get("executed_at"))
        and (read_at is None or visible_by(instant(row["executed_at"]), read_at))
        and row.get("origin") != "safety"
        and row.get("reason") not in ("normal_end", "safety_stop")
        and isinstance(row.get("parameters"), dict)
    ]
    return (
        max(valid, key=lambda row: (instant(row["executed_at"]), str(row.get("op_id", ""))))
        if valid
        else None
    )


def summarize_execution(session: dict, operations: list[dict], read_at: Any) -> list[dict]:
    unique = {}
    for row in operations:
        if row.get("session_id") == session.get("session_id") and row.get("op_id"):
            when = instant(row.get("executed_at"))
            if when and visible_by(when, read_at):
                unique[row["op_id"]] = row
    groups = {}
    channel_states = {}
    for row in unique.values():
        if _applied(row):
            channel = (row.get("phase"), row.get("node_index"))
            when = instant(row["executed_at"])
            channel_states.setdefault(channel, {}).setdefault(when, []).append(row)
    boundaries_by_op = {}
    session_end = instant(session.get("ended_at"))
    for states in channel_states.values():
        times = sorted(states)
        for index, start in enumerate(times):
            if len(states[start]) != 1:
                continue
            boundaries = times[index + 1 : index + 2]
            if session_end and start < session_end and visible_by(session_end, read_at):
                boundaries.append(session_end)
            if boundaries:
                boundaries_by_op[states[start][0]["op_id"]] = min(boundaries)
    for row in unique.values():
        key = (
            row.get("plan_version"),
            row.get("phase", ""),
            row.get("node_index"),
            row.get("resource_id", ""),
            row.get("resource_version", ""),
        )
        groups.setdefault(key, []).append(row)
    items = []
    for key, rows in groups.items():
        version, phase, node_index, resource, resource_version = key
        item = {
            "session_id": session["session_id"],
            "plan_id": session.get("plan_id", ""),
            "phase": phase,
            "resource_id": resource,
            "resource_version": resource_version,
            "command_count": len(rows),
            "ack_count": sum(bool(row.get("ack")) for row in rows),
            "failure_count": sum(row.get("success") is False for row in rows),
            "manual_select_count": 0,
            "manual_change_count": 0,
            "skip_count": 0,
            "stop_counts_by_reason": {},
            "source_refs": [],
        }
        if version is not None:
            item["plan_version"] = version
        if node_index is not None:
            item["node_index"] = node_index
        if session.get("episode_id"):
            item["episode_id"] = session["episode_id"]
        for row in rows:
            if row.get("origin") == "manual":
                counter = {
                    "select": "manual_select_count",
                    "change": "manual_change_count",
                    "skip": "skip_count",
                }.get(row.get("method"))
                if counter:
                    item[counter] += 1
                if row.get("method") == "stop":
                    reason = row.get("reason", "unknown")
                    item["stop_counts_by_reason"][reason] = (
                        item["stop_counts_by_reason"].get(reason, 0) + 1
                    )
        confirmed, unknown, bounded = 0.0, 0.0, False
        for row in rows:
            if not _applied(row):
                continue
            start = instant(row["executed_at"])
            end = boundaries_by_op.get(row["op_id"])
            if end is None:
                continue
            duration = (end - start).total_seconds()
            enabled = row.get("parameters", {}).get("enabled")
            if enabled is True:
                confirmed += duration
                bounded = True
            elif enabled is False:
                bounded = True
            else:
                unknown += duration
        if bounded:
            item["confirmed_on_sec"] = confirmed
        if unknown:
            item["unknown_sec"] = unknown
        last = select_last_effective_setting(rows, read_at)
        if last:
            item["last_confirmed_nonterminal_setting"] = {
                "parameters": last["parameters"],
                "occurred_at_utc": instant(last["executed_at"]).isoformat(),
                "event_ref": "operations:" + last["op_id"],
            }
        items.append(item)
    return items


def aggregate_detected_events(events: list[dict], monitored_sec: float | None) -> list[dict]:
    grouped = {}
    for row in {row["event_id"]: row for row in events if row.get("event_id")}.values():
        grouped.setdefault(row.get("type", "unknown"), []).append(row)
    result = []
    for kind, rows in sorted(grouped.items()):
        durations = [row.get("duration_sec") for row in rows]
        item = {"type": kind, "count": len(rows)}
        if all(finite_number(value) and value >= 0 for value in durations):
            item["duration_sec"] = sum(durations)
        if finite_number(monitored_sec) and monitored_sec > 0:
            item["monitored_sec"] = monitored_sec
        result.append(item)
    return result


def compare_plan_execution(final: dict, operations: list[dict], read_at: Any) -> list[dict]:
    effective = deepcopy(final)
    applied = {}
    valid = [
        row
        for row in operations
        if _applied(row)
        and instant(row.get("executed_at"))
        and visible_by(instant(row["executed_at"]), read_at)
        and row.get("origin") != "safety"
        and row.get("reason") not in ("normal_end", "safety_stop")
    ]
    for row in sorted(valid, key=lambda row: (instant(row["executed_at"]), row.get("op_id", ""))):
        parameters = row.get("parameters", {})
        if not isinstance(parameters, dict) or not row.get("phase") or not row.get("node_key"):
            continue
        nodes = [
            node
            for phase in effective.get("phases", [])
            if phase.get("phase") == row["phase"]
            for node in phase.get("nodes", [])
            if node.get("node_key") == row["node_key"]
        ]
        if len(nodes) != 1:
            continue
        for modality in ("sound", "light", "scent"):
            if isinstance(parameters.get(modality), dict):
                prefix = (
                    f"/phases/{_escape(row['phase'])}/nodes/{_escape(row['node_key'])}/{modality}"
                )

                def merge(target: dict, patch: dict, pointer: str) -> None:
                    for name, value in patch.items():
                        path = pointer + "/" + _escape(name)
                        if isinstance(value, dict) and isinstance(target.get(name), dict):
                            merge(target[name], value, path)
                        else:
                            target[name] = deepcopy(value)
                            applied[path] = row

                merge(nodes[0].setdefault(modality, {}), parameters[modality], prefix)
    changes = diff_plan_versions(final, effective)
    for change in changes:
        for prefix, row in applied.items():
            if change["path"] == prefix or change["path"].startswith(prefix + "/"):
                change["origins"] = [
                    row.get("origin")
                    if row.get("origin") in ("manual", "automation", "safety")
                    else "unknown"
                ]
                change["event_refs"] = ["operations:" + str(row.get("op_id", ""))]
    return changes
