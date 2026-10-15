from __future__ import annotations

from datetime import datetime
from statistics import mean

from app.server.somni.facts.calc import finite_number
from app.server.somni.facts.models import section
from app.server.somni.facts.time import attribute_sleep_date, instant, parse_date

POLICY = "sleep_debt_14_valid_days.v1"
REQUIRED_DAYS = 14
FALLBACK_MIN = 450.0


def eligible_sleep(row: dict, cutoff: datetime | None = None) -> bool:
    sleep = row.get("sleep")
    quality = row.get("quality")
    if not isinstance(sleep, dict) or not isinstance(quality, dict):
        return False
    value = sleep.get("total_asleep_min")
    start, end = instant(sleep.get("start_at")), instant(sleep.get("end_at"))
    received = instant(row.get("received_at"))
    try:
        wake_date = attribute_sleep_date(end, row.get("timezone_offset", "")) if end else None
    except (TypeError, ValueError):
        return False
    return bool(
        finite_number(value)
        and value >= 0
        and quality.get("valid") is True
        and row.get("main_sleep") is True
        and row.get("selection_policy_version")
        and row.get("source")
        and row.get("device_algorithm")
        and row.get("definition_version")
        and start
        and end
        and start < end
        and (cutoff is None or end <= cutoff)
        and received
        and (cutoff is None or received <= cutoff)
        and wake_date == row.get("record_date")
        and sleep.get("unit", "min") == "min"
        and sleep.get("method", "source_daily") == "source_daily"
    )


def cohort(row: dict) -> tuple:
    return tuple(
        row.get(key)
        for key in ("source", "device_algorithm", "definition_version", "selection_policy_version")
    )


def select_daily(rows: list[dict], cutoff: datetime | None = None) -> dict[str, dict]:
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        day = row.get("record_date")
        try:
            parse_date(day)
        except (TypeError, ValueError):
            continue
        received = instant(row.get("received_at"))
        if received and (cutoff is None or received <= cutoff):
            by_date.setdefault(day, []).append(row)
    selected = {}
    for day, candidates in by_date.items():
        if not all(
            isinstance(row.get("revision"), int)
            and not isinstance(row["revision"], bool)
            and row["revision"] >= 1
            and isinstance(row.get("episode_id"), str)
            and row["episode_id"]
            and isinstance(row.get("source"), str)
            and row["source"]
            for row in candidates
        ):
            continue
        episodes: dict[tuple, list[dict]] = {}
        for row in candidates:
            episodes.setdefault((row["source"], row["episode_id"]), []).append(row)
        main = []
        ambiguous = False
        for versions in episodes.values():
            revision = max(row["revision"] for row in versions)
            latest = [row for row in versions if row["revision"] == revision]
            if len(latest) != 1 or latest[0].get("main_sleep") not in (True, False):
                ambiguous = True
                break
            if latest[0].get("main_sleep") is True:
                main.append(latest[0])
        if not ambiguous and len(main) == 1 and eligible_sleep(main[0], cutoff):
            selected[day] = main[0]
    return selected


def calculate_sleep_debt(
    rows: list[dict], record_date: str, cutoff: datetime | None = None
) -> dict:
    parse_date(record_date)
    daily = select_daily(rows, cutoff)
    current = daily.get(record_date)
    value = {
        "record_date": record_date,
        "baseline_required_valid_days": REQUIRED_DAYS,
        "baseline_valid_day_count": 0,
        "baseline_policy_version": POLICY,
        "target_source": "unknown",
    }
    if current is None:
        return section(value, "partial", ["current_main_sleep_missing_or_ambiguous"])
    history = [
        row
        for day, row in sorted(daily.items(), reverse=True)
        if day < record_date and cohort(row) == cohort(current)
    ][:REQUIRED_DAYS]
    value.update(
        actual_sleep_min=current["sleep"]["total_asleep_min"],
        episode_id=current["episode_id"],
        source=current["source"],
        device_algorithm=current["device_algorithm"],
        definition_version=current["definition_version"],
        baseline_valid_day_count=len(history),
    )
    if len(history) == REQUIRED_DAYS:
        target = mean(row["sleep"]["total_asleep_min"] for row in history)
        value.update(baseline_mean_min=target, target_source="personal_14_valid_days")
    else:
        target = FALLBACK_MIN
        value["target_source"] = "fallback_450_min"
    value.update(
        target_sleep_min=target, daily_debt_min=max(0.0, target - value["actual_sleep_min"])
    )
    if history:
        value.update(
            baseline_first_local_date=history[-1]["record_date"],
            baseline_last_local_date=history[0]["record_date"],
        )
    value["source_refs"] = sorted(
        {ref for row in [current, *history] for ref in row.get("source_refs", [])}
    )
    return section(
        value,
        "ok" if len(history) == REQUIRED_DAYS else "partial",
        [] if len(history) == REQUIRED_DAYS else ["baseline_insufficient_fallback_450_min"],
    )
