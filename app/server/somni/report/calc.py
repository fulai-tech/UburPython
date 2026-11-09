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


# ---- session payload / final_stages（本地墙钟毫秒戳，不做 UTC 换算）----

STAGE_AWAKE = 1
STAGE_LIGHT = 2
STAGE_DEEP = 3
STAGE_REM = 4

_STAGE_NAME = {
    STAGE_AWAKE: "AWAKE",
    STAGE_LIGHT: "LIGHT",
    STAGE_DEEP: "DEEP",
    STAGE_REM: "REM",
}


def as_ms(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def ms_to_minutes(ms: int) -> int:
    if ms <= 0:
        return 0
    return math.floor(ms / 60_000)


def format_local_ms_hhmm(ms: int) -> str:
    """把本地墙钟毫秒戳直接格式化为 HH:mm（按 UTC 日历字段，不做时区换算）。"""
    if ms <= 0:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%H:%M")


def _on_bed_segment_ms(item: Any) -> tuple[int, int, int]:
    """解析单段 on_bed，返回 (from, to, duration_ms)；非法返回 (0, 0, 0)。"""
    if not isinstance(item, dict):
        return 0, 0, 0
    start = as_ms(item.get("from"))
    end = as_ms(item.get("to"))
    if start <= 0 or end <= start:
        return 0, 0, 0
    duration = as_ms(item.get("duration_ms"))
    if duration <= 0:
        duration = end - start
    return start, end, duration


def on_bed_bounds(on_bed: Any) -> tuple[int, int]:
    """取第一段 from 与最后一段 to；非法返回 (0, 0)。"""
    if not isinstance(on_bed, list) or not on_bed:
        return 0, 0
    first_from = 0
    last_to = 0
    for item in on_bed:
        start, end, _duration = _on_bed_segment_ms(item)
        if start <= 0:
            continue
        if first_from <= 0:
            first_from = start
        last_to = end
    if first_from <= 0 or last_to <= first_from:
        return 0, 0
    return first_from, last_to


def on_bed_duration_ms(on_bed: Any) -> int:
    """多段 on_bed 原始时长求和（不含离床间隙）。"""
    if not isinstance(on_bed, list):
        return 0
    total = 0
    for item in on_bed:
        _start, _end, duration = _on_bed_segment_ms(item)
        total += duration
    return total


def normalize_segments(
    on_bed: Any,
    segments: Any,
) -> list[dict[str, Any]]:
    """归一 segments：补 start_ts_ms；若首段 on_bed.from 早于首段则前置 AWAKE。"""
    bed_from, _bed_to = on_bed_bounds(on_bed)
    raw_list = segments if isinstance(segments, list) else []
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        stage = as_int(item.get("stage"), default=0)
        duration = as_ms(item.get("duration_ms"))
        start = as_ms(item.get("start_ts_ms"))
        if start <= 0:
            start = cursor if cursor > 0 else bed_from
        if duration < 0:
            duration = 0
        name = str(item.get("stage_name") or _STAGE_NAME.get(stage, "")).split("/", 1)[0]
        normalized.append(
            {
                "stage": stage,
                "stage_name": name or _STAGE_NAME.get(stage, ""),
                "duration_ms": duration,
                "start_ts_ms": start,
            }
        )
        cursor = start + duration

    if not normalized:
        return []

    first_start = as_ms(normalized[0].get("start_ts_ms"))
    if bed_from > 0 and first_start > bed_from:
        gap = first_start - bed_from
        normalized.insert(
            0,
            {
                "stage": STAGE_AWAKE,
                "stage_name": "AWAKE",
                "duration_ms": gap,
                "start_ts_ms": bed_from,
            },
        )
    return normalized


def sleep_structure_from_segments(
    on_bed: Any,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    tib_ms = on_bed_duration_ms(on_bed)
    tib_min = ms_to_minutes(tib_ms)
    stage_ms = {
        "awake": 0,
        "rem_sleep": 0,
        "light_sleep": 0,
        "deep_sleep": 0,
    }
    for seg in segments:
        duration = as_ms(seg.get("duration_ms"))
        stage = as_int(seg.get("stage"))
        if stage == STAGE_AWAKE:
            stage_ms["awake"] += duration
        elif stage == STAGE_REM:
            stage_ms["rem_sleep"] += duration
        elif stage == STAGE_LIGHT:
            stage_ms["light_sleep"] += duration
        elif stage == STAGE_DEEP:
            stage_ms["deep_sleep"] += duration

    sleep_ms = (
        stage_ms["rem_sleep"] + stage_ms["light_sleep"] + stage_ms["deep_sleep"]
    )
    mins = {key: ms_to_minutes(value) for key, value in stage_ms.items()}

    def part(key: str) -> dict[str, int]:
        m = mins[key]
        pct = math.floor(m / tib_min * 100) if tib_min > 0 else 0
        return {"minutes": m, "percent": pct}

    return {
        "awake": part("awake"),
        "rem_sleep": part("rem_sleep"),
        "light_sleep": part("light_sleep"),
        "deep_sleep": part("deep_sleep"),
        "total_minutes": ms_to_minutes(sleep_ms),
        "tib_minutes": tib_min,
    }


def sleep_quality_from_session(
    on_bed: Any,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    bed_from, bed_to = on_bed_bounds(on_bed)
    structure = sleep_structure_from_segments(on_bed, segments)
    tib = int(structure["tib_minutes"])
    sleep_min = int(structure["total_minutes"])
    efficiency = math.floor(sleep_min / tib * 100) if tib > 0 else 0

    sol = 0
    for seg in segments:
        if as_int(seg.get("stage")) != STAGE_AWAKE:
            sol = ms_to_minutes(as_ms(seg.get("start_ts_ms")) - bed_from)
            break

    # WASO = 最后一条 on_bed.to − 最后一个清醒阶段 start
    waso = 0
    last_awake_start = 0
    for seg in segments:
        if as_int(seg.get("stage")) == STAGE_AWAKE:
            last_awake_start = as_ms(seg.get("start_ts_ms"))
    if last_awake_start > 0 and bed_to > last_awake_start:
        waso = ms_to_minutes(bed_to - last_awake_start)

    return {
        "time_in_bed_minutes": tib,
        "sleep_onset_latency_minutes": max(0, sol),
        "sleep_efficiency": max(0, min(100, efficiency)),
        "bedtime": format_local_ms_hhmm(bed_from),
        "wake_up_time": format_local_ms_hhmm(bed_to),
        "awake_after_onset_minutes": max(0, waso),
    }


def idf_from_segments(
    segments: list[dict[str, Any]],
    on_bed: Any = None,
) -> list[dict[str, str]]:
    """把 segments 转成 idf 行；末段 end 若早于最后一条 on_bed.to 则补齐（仅展示）。"""
    _, bed_to = on_bed_bounds(on_bed)
    rows: list[dict[str, str]] = []
    last_i = len(segments) - 1
    for i, seg in enumerate(segments):
        start = as_ms(seg.get("start_ts_ms"))
        end = start + as_ms(seg.get("duration_ms"))
        if i == last_i and bed_to > end:
            end = bed_to
        stage = str(seg.get("stage_name") or _STAGE_NAME.get(as_int(seg.get("stage")), ""))
        rows.append(
            {
                "stage": stage.lower() if stage else "",
                "start": format_local_ms_hhmm(start),
                "end": format_local_ms_hhmm(end),
            }
        )
    return rows


def session_body(payload_doc: dict[str, Any] | None) -> dict[str, Any]:
    """取出 session 正文：优先 payload 嵌套，否则文档顶层。"""
    if not payload_doc:
        return {}
    nested = payload_doc.get("payload")
    if isinstance(nested, dict):
        return nested
    return payload_doc


def final_stages_segments(body: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    on_bed = body.get("on_bed")
    final_stages = body.get("final_stages")
    segments_raw: Any = []
    if isinstance(final_stages, dict):
        segments_raw = final_stages.get("segments")
    return on_bed, normalize_segments(on_bed, segments_raw)
