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


def coerce_datetime(value: Any) -> datetime | None:
    """接受 datetime / ISO8601 字符串；非法则 None（不抛）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def format_hhmm(value: Any) -> str:
    dt = coerce_datetime(value)
    if dt is None:
        return ""
    return to_report_tz(dt).strftime("%H:%M")


def format_collected_at(value: Any) -> str:
    dt = coerce_datetime(value)
    if dt is None:
        return ""
    return to_report_tz(dt).isoformat()


def sleep_stage_parts(raw: dict[str, Any]) -> dict[str, Any]:
    bed = minutes_between(
        coerce_datetime(raw.get("bed_time")),
        coerce_datetime(raw.get("wake_up_time")),
    )
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


def resolve_report_timezone_name(timezone_name: str) -> str:
    """返回实际使用的 IANA 名；空或非法回退 Asia/Shanghai。"""
    name = (timezone_name or "").strip() or "Asia/Shanghai"
    try:
        ZoneInfo(name)
        return name
    except Exception:
        return "Asia/Shanghai"


def resolve_report_timezone(timezone_name: str) -> ZoneInfo:
    return ZoneInfo(resolve_report_timezone_name(timezone_name))


def local_today(timezone_name: str, now_utc: datetime | None = None) -> date:
    now = now_utc or datetime.now(UTC)
    return ensure_aware(now).astimezone(resolve_report_timezone(timezone_name)).date()


def trailing_week_bounds(end: date) -> tuple[str, str]:
    """以 end 为右端点的 7 个日历日（含 end）：[end-6, end]。

    睡眠按起床日入库：日历周一晚～周日晚 → 库中周二～下周一。
    每周一生成「上周」周报时，end 一般为当天周一，即 [上周二, 本周一]。
    """
    start = end - timedelta(days=6)
    return start.isoformat(), end.isoformat()


def default_week_bounds(
    *,
    timezone_name: str = "",
    now_utc: datetime | None = None,
) -> tuple[str, str]:
    """无 start/end 时：以用户时区本地今天为右端，取近 7 个起床日。"""
    return trailing_week_bounds(local_today(timezone_name, now_utc=now_utc))


def baseline_start(end_date: str) -> str:
    return (parse_record_date(end_date) - timedelta(days=13)).isoformat()


def empty_weekly_day_metric(record_date: str) -> dict[str, Any]:
    return {
        "record_date": record_date,
        "bed_time": "",
        "wake_up_time": "",
        "total_sleep_minutes": 0,
        "sleep_efficiency": 0,
        "rem_percent": 0,
        "deep_sleep_percent": 0,
    }


def weekly_day_metric(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("raw_data") if isinstance(record.get("raw_data"), dict) else {}
    raw = raw or {}
    parts = sleep_stage_parts(raw)
    return {
        "record_date": str(record.get("record_date") or ""),
        "bed_time": format_collected_at(raw.get("bed_time")),
        "wake_up_time": format_collected_at(raw.get("wake_up_time")),
        "total_sleep_minutes": int(parts["total_minutes"]),
        "sleep_efficiency": as_int(raw.get("sleep_efficiency")),
        "rem_percent": as_int(raw.get("rem_ratio")),
        "deep_sleep_percent": as_int(raw.get("deep_sleep_ratio")),
    }


def iter_date_range(start_date: str, end_date: str) -> list[str]:
    """闭区间 [start_date, end_date] 的 YYYY-MM-DD 列表（升序）。"""
    start = parse_record_date(start_date)
    end = parse_record_date(end_date)
    if end < start:
        return []
    days: list[str] = []
    cur = start
    while cur <= end:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def align_weekly_days(
    metrics_by_date: dict[str, dict[str, Any]],
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """按起床日对齐填充；缺日补空点（时间为空串、数值为 0），保证序列连续、长度固定。"""
    aligned: list[dict[str, Any]] = []
    for day in iter_date_range(start_date, end_date):
        hit = metrics_by_date.get(day)
        aligned.append(dict(hit) if hit else empty_weekly_day_metric(day))
    return aligned
