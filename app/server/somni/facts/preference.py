"""Mechanical preference evidence aggregation; policy thresholds belong to DeepHealth."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from math import isfinite
from typing import Any

from app.server.somni.facts.models import timestamp
from app.server.somni.facts.time import instant, parse_date, parse_offset

PARAMETERS = {"volume": (0, 100), "brightness": (0, 100), "color_temp": (1800, 6500)}
PHASES = {"Unwind", "Soothe", "Guard", "Wake"}


def number(value: Any, parameter: str) -> bool:
    """Reject booleans, nulls, nonfinite values and values outside device units."""
    lower, upper = PARAMETERS[parameter]
    return type(value) in (int, float) and isfinite(value) and lower <= value <= upper


def _ref(row: dict, collection: str) -> str:
    key = row.get("op_id") or row.get("snapshot_id") or row.get("session_id") or row.get("_id")
    return f"{collection}:{key}" if key else ""


def _parameters(snapshot: dict) -> dict[tuple[str, str, str, str], float]:
    result = {}
    for phase in snapshot.get("phases", []):
        name = phase.get("phase") or phase.get("name") or phase.get("type")
        if name not in PHASES:
            continue
        for index, node in enumerate(phase.get("nodes", [])):
            key = str(node.get("node_type") or node.get("node_key") or index)
            light = node.get("light") or {}
            # Disabled-light remembered settings and curves are not scalar preferences.
            if light.get("enabled") is True and light.get("mode") == "solid":
                for parameter in ("brightness", "color_temp"):
                    if number(light.get(parameter), parameter):
                        result[name, key, "", parameter] = light[parameter]
            for track in (node.get("sound") or {}).get("tracks", []):
                resource = track.get("audio_id")
                if resource and number(track.get("volume"), "volume"):
                    result[name, key, str(resource), "volume"] = track["volume"]
    return result


def merge_operation_bursts(rows: list[dict]) -> list[dict]:
    """Deduplicate retransmissions, then retain endpoints of bounded knob gestures."""
    unique = {}
    segment = defaultdict(int)
    previous_signature = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            instant(row.get("executed_at")) or datetime.min.replace(tzinfo=parse_offset("+00:00"))
        ),
    )
    for row in ordered:
        occurred = instant(row.get("executed_at"))
        if occurred is None or occurred.year < 2020:
            continue
        applied = row.get("applied")
        applied = applied if isinstance(applied, dict) else {}
        payload = row.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        device = str(row.get("device_id") or row.get("sn") or "")
        session_id = str(row.get("session_id") or applied.get("session_id") or "")
        stream = (str(row.get("uid") or ""), device, session_id)
        signature = (
            row.get("method"),
            row.get("origin"),
            row.get("phase") or applied.get("phase"),
            row.get("node_key") or applied.get("node_key"),
            row.get("resource_id") or applied.get("resource_id"),
            applied.get("on", payload.get("on")),
        )
        if previous_signature.get(stream) != signature:
            segment[stream] += 1
            previous_signature[stream] = signature
        # Explicit APPLIED is required for canonical sources. Legacy successful
        # local telemetry may be observed, but never establishes hold proof.
        if row.get("success") is not True or (
            applied.get("execution_state") != "APPLIED" and not row.get("_storage_telemetry_only")
        ):
            continue
        owner = applied.get("owner") or row.get("owner")
        if owner and owner not in ("PLAN", "plan"):
            continue
        method = row.get("method")
        values = {}
        if method == "sound.set":
            values["volume"] = applied.get("value", applied.get("volume"))
        elif method == "nightlight.set":
            if applied.get("on", payload.get("on")) is False:
                continue
            values = {key: applied.get(key) for key in ("brightness", "color_temp")}
        for parameter, value in values.items():
            if not number(value, parameter):
                continue
            session = row.get("session_id") or applied.get("session_id") or ""
            group = (
                str(row.get("uid") or ""),
                str(row.get("device_id") or row.get("sn") or ""),
                str(session),
                str(row.get("plan_id") or applied.get("plan_id") or ""),
                row.get("plan_version", applied.get("plan_version")),
                str(row.get("phase") or applied.get("phase") or ""),
                str(row.get("node_key") or applied.get("node_key") or ""),
                str(row.get("resource_id") or applied.get("resource_id") or ""),
                parameter,
                str(row.get("origin") or "unknown"),
                segment[stream],
            )
            if not group[0] or not group[1]:
                continue
            # Exact content keeps different steps sharing a request distinct.
            identity = (group, occurred, value)
            event = unique.setdefault(
                identity, {"group": group, "at": occurred, "value": value, "refs": []}
            )
            ref = _ref(row, "operations")
            if ref:
                event["refs"].append(ref)
            duration = (
                None
                if row.get("_storage_telemetry_only")
                else applied.get("confirmed_duration_sec")
            )
            event["hold_verified"] = (
                type(duration) in (int, float) and isfinite(duration) and duration >= 120
            )
    grouped = defaultdict(list)
    for event in unique.values():
        grouped[event["group"]].append(event)
    result = []
    for group, events in sorted(grouped.items(), key=lambda item: str(item[0])):
        values_at = defaultdict(set)
        for event in events:
            values_at[event["at"]].add(event["value"])
        events.sort(key=lambda event: (event["at"], event["value"]))
        bursts = []
        for event in events:
            if (
                not bursts
                or event["at"] - bursts[-1][-1]["at"] > timedelta(seconds=2)
                or event["at"] - bursts[-1][0]["at"] > timedelta(seconds=30)
            ):
                bursts.append([])
            bursts[-1].append(event)
        for events_in_burst in bursts:
            first, last = events_in_burst[0], events_in_burst[-1]
            changes = [
                b["value"] - a["value"] for a, b in zip(events_in_burst, events_in_burst[1:])
            ]
            result.append(
                {
                    "uid": group[0],
                    "device_id": group[1],
                    "session_id": group[2],
                    "plan_id": group[3],
                    "plan_version": group[4],
                    "phase": group[5],
                    "node_key": group[6],
                    "resource_id": group[7],
                    "parameter": group[8],
                    "origin": group[9],
                    "first_at_utc": timestamp(first["at"]),
                    "last_at_utc": timestamp(last["at"]),
                    "first_value": first["value"],
                    "last_value": last["value"],
                    "step_count": len(events_in_burst),
                    "hold_verified": last.get("hold_verified", False),
                    "ambiguous_order": any(
                        len(values_at[event["at"]]) > 1 for event in events_in_burst
                    ),
                    "reversed": any(value > 0 for value in changes)
                    and any(value < 0 for value in changes),
                    "source_refs": sorted(
                        {min(event["refs"]) for event in (first, last) if event["refs"]}
                    ),
                }
            )
    return result


def aggregate_preference_evidence(
    uid: str,
    end_local_date: str,
    timezone_offset: str,
    operations: list[dict],
    snapshots: list[dict],
    sessions: list[dict],
    *,
    read_at: datetime,
    phases: set[str] | None = None,
) -> dict:
    """Return facts with exact joins and explicit unknowns, never learned preferences.

    Night identity is the local date of session start. A complete exposure requires
    an ended, fully observed session and an exact owned snapshot. Saved revisions
    are evidence of intent only; adoption must be proved independently.
    """
    from app.server.somni.facts.preference_storage import normalize_sources

    # Demo collections are isolated; reject synthetic rows even if misrouted.
    snapshots = [r for r in snapshots if not r.get("synthetic") and not r.get("preview_only")]
    sessions = [
        r
        for r in sessions
        if not r.get("synthetic") and not r.get("previewing") and not r.get("preview_only")
    ]
    operations = [r for r in operations if not r.get("synthetic") and not r.get("preview_only")]
    snapshots, sessions, operations = normalize_sources(
        uid, timezone_offset, snapshots, sessions, operations
    )
    zone = parse_offset(timezone_offset)
    end = parse_date(end_local_date)
    start = end - timedelta(days=14)
    reasons = set()
    by_snapshot = defaultdict(list)
    for row in snapshots:
        lifecycle = row.get("lifecycle") or row.get("status")
        lifecycle = lifecycle.get("status") if isinstance(lifecycle, dict) else lifecycle
        if lifecycle not in {"active", "superseded", "committed"}:
            reasons.add("snapshot_lifecycle_unverified")
            continue
        if row.get("uid") == uid and type(row.get("version")) is int:
            by_snapshot[row.get("plan_id"), row["version"]].append(row)
    indexed = {key: rows[0] for key, rows in by_snapshot.items() if len(rows) == 1}
    if len(indexed) != len(by_snapshot):
        reasons.add("snapshot_revision_ambiguous")
    valid_sessions = {}
    exposures = []
    observations = []
    ongoing = False
    for session in sessions:
        if session.get("uid") != uid:
            continue
        began, ended = instant(session.get("started_at")), instant(session.get("ended_at"))
        if began is None:
            reasons.add("session_clock_unverified")
            continue
        snapshot = indexed.get((session.get("plan_id"), session.get("plan_version")))
        try:
            night = (
                parse_date(snapshot["plan_start_local_date"])
                if snapshot
                else began.astimezone(zone).date()
            )
        except (KeyError, TypeError, ValueError):
            reasons.add("session_business_date_mismatch")
            continue
        if night == end and began <= read_at and ended is None:
            ongoing = True
        if not start <= night < end:
            continue
        if ended is None:
            ongoing = True
            continue
        if ended <= began or ended > read_at:
            reasons.add("session_clock_unverified")
            continue
        if read_at - ended < timedelta(minutes=30):
            ongoing = True
            continue
        identifier = session.get("session_id")
        version = session.get("plan_version")
        snapshot = indexed.get((session.get("plan_id"), version))
        if not identifier or not snapshot or session.get("evidence_complete") is not True:
            reasons.add("session_or_snapshot_coverage_incomplete")
            continue
        if snapshot.get("plan_start_local_date") != night.isoformat():
            reasons.add("session_business_date_mismatch")
            continue
        if snapshot.get("timezone_offset") != timezone_offset:
            reasons.add("historical_timezone_unverified")
            continue
        if identifier in valid_sessions:
            reasons.add("session_identity_ambiguous")
            valid_sessions[identifier] = None
            continue
        valid_sessions[identifier] = (session, snapshot, night.isoformat())
    for context in valid_sessions.values():
        if context is None:
            continue
        session, snapshot, night = context
        executed_nodes = session.get("executed_nodes")
        completed = {
            (r.get("phase"), str(r.get("node_key")))
            for r in (executed_nodes or [])
            if isinstance(r, dict) and r.get("completed") is True
        }
        stage_maxima = {}
        for phase in snapshot.get("phases", []):
            name = phase.get("phase")
            required = {
                (name, str(n.get("node_type") or n.get("node_key") or i))
                for i, n in enumerate(phase.get("nodes", []))
            }
            values = [
                v
                for (p, _n, _r, parameter), v in _parameters(snapshot).items()
                if p == name and parameter == "volume"
            ]
            if required and required <= completed and values:
                stage_maxima[name] = max(values)
        for key in _parameters(snapshot):
            if phases is not None and key[0] not in phases:
                continue
            if not isinstance(executed_nodes, list) or not any(
                item.get("phase") == key[0]
                and str(item.get("node_key")) == key[1]
                and item.get("completed") is True
                for item in executed_nodes
                if isinstance(item, dict)
            ):
                reasons.add("node_exposure_unverified")
                continue
            phase, node, resource, parameter = key
            exposures.append(
                {
                    "night": night,
                    "started_at_utc": timestamp(instant(session["started_at"])),
                    "ended_at_utc": timestamp(instant(session["ended_at"])),
                    "phase": phase,
                    "node_key": node,
                    "resource_id": resource,
                    "parameter": parameter,
                    "source_refs": [_ref(session, "sessions"), _ref(snapshot, "snapshots")],
                    **(
                        {"stage_max_volume": stage_maxima[phase]}
                        if parameter == "volume" and phase in stage_maxima
                        else {}
                    ),
                }
            )
        # Only direct, ordered, explicitly manual revisions with unchanged node identity.
        revisions = sorted(
            (
                row
                for row in indexed.values()
                if row.get("uid") == uid
                and row.get("plan_id") == snapshot.get("plan_id")
                and type(row.get("version")) is int
            ),
            key=lambda row: row["version"],
        )
        for final in revisions:
            if indexed.get((final.get("plan_id"), final["version"])) is not final:
                continue
            original = indexed.get((final.get("plan_id"), final.get("base_plan_version")))
            created = instant(final.get("created_at"))
            parent_created = instant((original or {}).get("created_at"))
            if parent_created is not None and created is not None and parent_created >= created:
                continue
            if not original or final.get("edit_source") != "manual" or created is None:
                continue
            if (
                final.get("plan_start_local_date") != night
                or original.get("plan_start_local_date") != night
                or created > instant(session["ended_at"])
                or not _ref(original, "snapshots")
                or not _ref(final, "snapshots")
            ):
                continue
            if final["version"] <= original["version"]:
                continue
            adopted = (
                final["version"] == session.get("plan_version")
                and created <= instant(session["started_at"])
            ) or any(
                item.get("version") == final["version"]
                and instant(item.get("applied_at")) is not None
                and max(created, instant(session["started_at"]))
                <= instant(item["applied_at"])
                <= instant(session["ended_at"])
                for item in session.get("adopted_versions", [])
                if isinstance(item, dict)
            )

            def node_ids(row):
                return {
                    (
                        phase.get("phase"),
                        str(node.get("node_type") or node.get("node_key") or index),
                    ): node.get("node_id")
                    for phase in row.get("phases", [])
                    for index, node in enumerate(phase.get("nodes", []))
                }

            original_ids, final_ids = node_ids(original), node_ids(final)
            before, after = _parameters(original), _parameters(final)
            for key in before.keys() & after.keys():
                if phases is not None and key[0] not in phases:
                    continue
                if before[key] == after[key]:
                    continue
                if original_ids.get(key[:2]) != final_ids.get(key[:2]):
                    continue
                phase, node, resource, parameter = key
                observations.append(
                    {
                        "night": night,
                        "phase": phase,
                        "node_key": node,
                        "resource_id": resource,
                        "parameter": parameter,
                        "before": before[key],
                        "after": after[key],
                        "intent": "manual",
                        "applied": adopted,
                        "occurred_at_utc": timestamp(created),
                        "source": "snapshot",
                        "source_refs": [_ref(original, "snapshots"), _ref(final, "snapshots")],
                    }
                )
    bursts = merge_operation_bursts([row for row in operations if row.get("uid") == uid])
    for burst in bursts:
        context = valid_sessions.get(burst["session_id"])
        if context is None:
            reasons.add("operation_session_unverified")
            continue
        session, snapshot, night = context
        at = instant(burst["last_at_utc"])
        key = tuple(burst[field] for field in ("phase", "node_key", "resource_id", "parameter"))
        if phases is not None and key[0] not in phases:
            continue
        baseline = _parameters(snapshot).get(key)
        if (
            burst["device_id"] != session.get("device_id")
            or burst["plan_id"] != session.get("plan_id")
            or type(burst["plan_version"]) is not int
            or burst["plan_version"] != session.get("plan_version")
            or session.get("adopted_versions")
            or baseline is None
            or burst["origin"] != "local"
            or burst["ambiguous_order"]
            or not instant(session["started_at"])
            <= instant(burst["first_at_utc"])
            <= at
            <= instant(session["ended_at"])
        ):
            reasons.add("operation_plan_context_unverified")
            continue
        # No inference of hold duration from silence. Only confirmed manual plan adoption
        # qualifies automatically; hardware endpoints remain provisional observations.
        observations.append(
            {
                "night": night,
                "phase": key[0],
                "node_key": key[1],
                "resource_id": key[2],
                "parameter": key[3],
                "before": baseline,
                "after": burst["last_value"],
                "intent": "manual",
                "applied": True,
                "hold_verified": burst["hold_verified"],
                "reversed": burst["reversed"],
                "source": "operation",
                "occurred_at_utc": burst["last_at_utc"],
                "source_refs": sorted(
                    set(
                        [
                            *burst["source_refs"],
                            _ref(session, "sessions"),
                            _ref(snapshot, "snapshots"),
                        ]
                    )
                    - {""}
                ),
            }
        )
    return {
        "schema_version": "somni_preference_evidence.v1",
        "uid": uid,
        "review_end_local_date": end.isoformat(),
        "timezone_offset": timezone_offset,
        "start_local_date": start.isoformat(),
        "observations": observations,
        "exposures": exposures,
        "operation_bursts": bursts,
        "deferred": ongoing,
        "missing_reasons": sorted(reasons),
    }
