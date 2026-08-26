"""量产报告纯计算：本地日窗口、卧床/阶段分钟、均值 floor。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

REPORT_TZ = ZoneInfo("Asia/Shanghai")


def parse_record_date(record_date: str) -> date:
    return date.fromisoformat(record_date)


def local_day_utc_range(record_date: str) -> tuple[datetime, datetime]:
    """record_date 本地日 [00:00, 次日 00:00) 转 UTC 感知时间。"""
    day = parse_record_date(record_date)
    start_local = datetime(day.year, day.month, day.day, tzinfo=REPORT_TZ)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def to_report_tz(value: datetime) -> datetime:
    return ensure_aware(value).astimezone(REPORT_TZ)


def minutes_between(start: datetime | None, end: datetime | None) -> int:
    if start is None or end is None:
        return 0
    delta = to_report_tz(end) - to_report_tz(start)
    return max(0, math.floor(delta.total_seconds() / 60))


def stage_minutes(bed_minutes: int, ratio: int | float | None) -> int:
    if bed_minutes <= 0 or ratio is None:
        return 0
    return math.floor(bed_minutes * float(ratio) / 100)


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def floor_avg(values: Iterable[float]) -> int:
    nums = list(values)
    if not nums:
        return 0
    return math.floor(sum(nums) / len(nums))


def floor_metric_stats(values: Iterable[float]) -> dict[str, int]:
    nums = list(values)
    if not nums:
        return {"value": 0, "min": 0, "max": 0}
    return {
        "value": math.floor(sum(nums) / len(nums)),
        "min": math.floor(min(nums)),
        "max": math.floor(max(nums)),
    }


def format_hhmm(value: datetime | None) -> str:
    if value is None:
        return ""
    return to_report_tz(value).strftime("%H:%M")


def format_collected_at(value: datetime | None) -> str:
    if value is None:
        return ""
    return to_report_tz(value).isoformat()


def sleep_stage_parts(raw: dict[str, Any]) -> dict[str, Any]:
    bed = minutes_between(raw.get("bed_time"), raw.get("wake_up_time"))
    awake_ratio = as_int(raw.get("awake_ratio"))
    rem_ratio = as_int(raw.get("rem_ratio"))
    light_ratio = as_int(raw.get("light_sleep_ratio"))
    deep_ratio = as_int(raw.get("deep_sleep_ratio"))
    awake = stage_minutes(bed, awake_ratio)
    rem = stage_minutes(bed, rem_ratio)
    light = stage_minutes(bed, light_ratio)
    deep = stage_minutes(bed, deep_ratio)
    return {
        "bed_minutes": bed,
        "awake": {"minutes": awake, "percent": awake_ratio},
        "rem_sleep": {"minutes": rem, "percent": rem_ratio},
        "light_sleep": {"minutes": light, "percent": light_ratio},
        "deep_sleep": {"minutes": deep, "percent": deep_ratio},
        "total_minutes": deep + light + rem,
    }
