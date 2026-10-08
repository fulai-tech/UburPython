from __future__ import annotations

import math
from copy import deepcopy
from datetime import timedelta
from statistics import mean, median
from typing import Any

from app.server.somni.facts.time import TimeScope, instant, parse_date


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


def build_series(metrics: list[dict], scope: TimeScope) -> list[dict]:
    grouped = {}
    fields = (
        "metric_id",
        "unit",
        "source",
        "device_algorithm",
        "method",
        "definition_version",
        "hrv_method",
    )
    for metric in metrics:
        if not scope.start(28).isoformat() <= metric["local_date"] < scope.end.isoformat():
            continue
        key = tuple(metric.get(field, "") for field in fields)
        grouped.setdefault(key, {}).setdefault(metric["local_date"], []).append(metric)
    result = []
    for key, days in sorted(grouped.items()):
        daily = []
        for local_date, candidates in sorted(days.items()):
            item = dict(candidates[0])
            if len(candidates) > 1:
                item.pop("value", None)
                item.update(quality="missing", missing_reason="ambiguous_daily_value")
            daily.append(item)
        indexed = {item["local_date"]: item for item in daily}
        series = {**dict(zip(fields, key, strict=True)), "daily": daily}
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
        and (read_at is None or instant(row["executed_at"]) <= read_at)
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
            if when and when <= read_at:
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
            if session_end and start < session_end <= read_at:
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
        and instant(row["executed_at"]) <= read_at
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
