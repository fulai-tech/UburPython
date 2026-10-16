"""Translate stored Mongo shapes into evidence, without writing derived fields back.

A legacy completed session proves exposure only when its exact owned revision,
device, timezone, start, duration and node count agree. Successful knob telemetry
is observable, but does not prove the chosen value was held.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.server.somni.facts.time import instant, parse_date, parse_offset

REJECTED = {"discarded", "device_rejected", "queued", "pending"}


def normalize_snapshot(row: dict) -> dict:
    """Map production date/time/source fields; reject explicit invalid states."""
    result = deepcopy(row)
    if "plan_start_local_date" in row:
        if row.get("apply_status") in REJECTED:
            result["lifecycle"] = "rejected"
            result["status"] = "rejected"
        return result
    if not isinstance(row.get("date"), str) or instant(row.get("create_time")) is None:
        return result
    status = row.get("status")
    if "status" in row and (type(status) is not int or status != 0):
        return result
    if row.get("apply_status") not in (None, "", "applied"):
        return result
    result.update(
        plan_start_local_date=row["date"],
        created_at=row["create_time"],
        lifecycle="committed",
        edit_source="manual" if row.get("edit_source") == "app" else row.get("edit_source"),
    )
    return result


def stored_phase_content(value):
    """Ignore only Mongo's optional null placeholders when comparing plan bodies."""
    if isinstance(value, dict):
        return {key: stored_phase_content(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [stored_phase_content(item) for item in value]
    return value


def node_intervals(snapshot: dict, zone) -> list[tuple]:
    """Build a contiguous schedule anchored to the stored business date."""
    result = []
    try:
        day = parse_date(snapshot["plan_start_local_date"])
        prior_end = None
        for phase in snapshot["phases"]:
            wall = datetime.strptime(
                phase["start_time"], "%H:%M:%S" if len(phase["start_time"]) == 8 else "%H:%M"
            ).time()
            start = datetime.combine(day, wall, zone)
            if prior_end is not None:
                while start < prior_end:
                    start += timedelta(days=1)
                if start != prior_end:
                    return []
            cursor = start
            nodes = phase["nodes"]
            if not isinstance(nodes, list) or not nodes:
                return []
            for index, node in enumerate(nodes):
                duration = node.get("duration_sec")
                if type(duration) is not int or duration <= 0:
                    return []
                end = cursor + timedelta(seconds=duration)
                result.append(
                    (
                        phase["phase"],
                        str(node.get("node_type") or node.get("node_key") or index),
                        cursor,
                        end,
                        node,
                    )
                )
                cursor = end
            if (
                type(phase.get("duration_sec")) is not int
                or (cursor - start).total_seconds() != phase["duration_sec"]
            ):
                return []
            prior_end = cursor
    except (KeyError, TypeError, ValueError, OverflowError):
        return []
    return result


def normalize_sources(
    uid: str, offset: str, snapshots: list[dict], sessions: list[dict], operations: list[dict]
) -> tuple[list, list, list]:
    """Normalize only relationships backed by unique owned persisted revisions."""
    zone = parse_offset(offset)
    normalized = [normalize_snapshot(row) for row in snapshots]
    counts = Counter(
        (row.get("plan_id"), row.get("version")) for row in normalized if row.get("uid") == uid
    )
    indexed = {
        (row.get("plan_id"), row.get("version")): row
        for row in normalized
        if row.get("uid") == uid and counts[row.get("plan_id"), row.get("version")] == 1
    }
    for source, row in zip(snapshots, normalized, strict=True):
        if row.get("uid") != uid or "plan_start_local_date" in source:
            continue
        prior = (
            indexed.get((row.get("plan_id"), row.get("version", 0) - 1))
            if type(row.get("version")) is int
            else None
        )
        before, after = instant((prior or {}).get("created_at")), instant(row.get("created_at"))
        if (
            prior
            and before
            and after
            and before < after
            and prior.get("plan_start_local_date") == row.get("plan_start_local_date")
        ):
            row["base_plan_version"] = prior["version"]
    normalized_sessions = []
    for source in sessions:
        row = deepcopy(source)
        if row.get("uid") not in (None, uid):
            normalized_sessions.append(row)
            continue
        snapshot = indexed.get((row.get("plan_id"), row.get("plan_version")))
        if not snapshot or snapshot.get("lifecycle") not in {"active", "committed", "superseded"}:
            normalized_sessions.append(row)
            continue
        if "uid" in row:  # Explicit canonical coverage remains authoritative.
            normalized_sessions.append(row)
            continue
        began, ended = instant(row.get("started_at")), instant(row.get("ended_at"))
        schedule = node_intervals(snapshot, zone)
        try:
            historical_zone = ZoneInfo(row.get("timezone") or "")
            timezone_ok = (
                began is not None
                and began.astimezone(historical_zone).utcoffset()
                == began.astimezone(zone).utcoffset()
                and ended is not None
                and ended.astimezone(historical_zone).utcoffset()
                == ended.astimezone(zone).utcoffset()
            )
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            timezone_ok = False
        # Do not fabricate completeness for stopped, failed, paused or preview sessions.
        complete = (
            row.get("state") == "ended"
            and row.get("end_reason") == "completed"
            and row.get("previewing") is False
            and timezone_ok
            and bool(schedule)
            and bool(row.get("device_id"))
            and row.get("device_id") == snapshot.get("device_id")
            and type(row.get("total_nodes")) is int
            and row["total_nodes"] == len(schedule)
            and instant(snapshot.get("created_at")) is not None
            and instant(snapshot["created_at"]) <= began
            and began == schedule[0][2]
            and ended is not None
            and ended >= schedule[-1][3]
            and ended <= schedule[-1][3] + timedelta(minutes=30)
        )
        if complete:
            row.update(
                uid=uid,
                evidence_complete=True,
                executed_nodes=[
                    {"phase": phase, "node_key": key, "completed": True}
                    for phase, key, *_ in schedule
                ],
            )
            snapshot["timezone_offset"] = offset
        normalized_sessions.append(row)
    normalized_ops = []
    for source in operations:
        row = deepcopy(source)
        applied = row.get("applied") if isinstance(row.get("applied"), dict) else {}
        if (
            applied.get("execution_state")
            or row.get("uid") != uid
            or row.get("origin") != "local"
            or row.get("success") is not True
        ):
            normalized_ops.append(row)
            continue
        at = instant(row.get("executed_at"))
        matches = []
        for session in normalized_sessions:
            if (
                session.get("uid") != uid
                or session.get("evidence_complete") is not True
                or row.get("device_id") != session.get("device_id")
            ):
                continue
            if row.get("session_id") and row["session_id"] != session.get("session_id"):
                continue
            snap = indexed.get((session.get("plan_id"), session.get("plan_version")))
            for phase, key, start, end, node in node_intervals(snap, zone):
                if at is not None and start <= at < end:
                    tracks = (node.get("sound") or {}).get("tracks") or []
                    resources = {track.get("audio_id") for track in tracks if track.get("audio_id")}
                    if row.get("method") == "sound.set" and len(resources) != 1:
                        continue
                    matches.append(
                        (
                            session,
                            phase,
                            key,
                            next(iter(resources)) if row.get("method") == "sound.set" else "",
                        )
                    )
        if len(matches) == 1:
            session, phase, key, resource = matches[0]
            # Explicit references may narrow matching, never be overwritten on conflict.
            expected = dict(
                session_id=session["session_id"],
                plan_id=session["plan_id"],
                plan_version=session["plan_version"],
                phase=phase,
                node_key=key,
                resource_id=resource,
            )
            if all(
                row.get(k) in (None, "", v) and applied.get(k) in (None, "", v)
                for k, v in expected.items()
            ):
                row.update(expected)
                row["applied"] = dict(applied)
                row["_storage_telemetry_only"] = True
        normalized_ops.append(row)
    return normalized, normalized_sessions, normalized_ops
