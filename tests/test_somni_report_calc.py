"""量产报告 calc 单测。"""

from __future__ import annotations

from datetime import UTC, date, datetime

from app.server.somni.report.calc import (
    align_weekly_days,
    baseline_start,
    coerce_datetime,
    default_week_bounds,
    floor_avg,
    floor_metric_stats,
    format_collected_at,
    format_hhmm,
    local_day_utc_range,
    minutes_between,
    resolve_report_timezone_name,
    sleep_stage_parts,
    stage_minutes,
    trailing_week_bounds,
    weekly_day_metric,
)


def test_local_day_utc_range_shanghai() -> None:
    start, end = local_day_utc_range("2026-08-10")
    assert start == datetime(2026, 8, 9, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 8, 10, 16, 0, tzinfo=UTC)


def test_minutes_between_and_stage() -> None:
    bed = datetime(2026, 8, 9, 15, 30, tzinfo=UTC)  # 23:30 CST
    wake = datetime(2026, 8, 10, 0, 30, tzinfo=UTC)  # 08:30 CST
    assert minutes_between(bed, wake) == 540
    assert stage_minutes(540, 17) == 91
    assert stage_minutes(0, 17) == 0


def test_floor_avg_and_metric_stats() -> None:
    assert floor_avg([66.5, 62.2]) == 64
    assert floor_avg([]) == 0
    assert floor_metric_stats([21.9, 23.2, 25.8]) == {
        "value": 23,
        "min": 21,
        "max": 25,
    }
    assert floor_metric_stats([]) == {"value": 0, "min": 0, "max": 0}


def test_format_hhmm() -> None:
    assert format_hhmm(datetime(2026, 8, 9, 15, 30, tzinfo=UTC)) == "23:30"
    assert format_hhmm(None) == ""


def test_sleep_stage_parts_excludes_awake_from_total() -> None:
    raw = {
        "bed_time": datetime(2026, 8, 9, 15, 30, tzinfo=UTC),
        "wake_up_time": datetime(2026, 8, 10, 0, 30, tzinfo=UTC),
        "awake_ratio": 10,
        "deep_sleep_ratio": 20,
        "light_sleep_ratio": 50,
        "rem_ratio": 20,
    }
    parts = sleep_stage_parts(raw)
    assert parts["bed_minutes"] == 540
    assert parts["deep_sleep"]["minutes"] == 108
    assert parts["total_minutes"] == 108 + 270 + 108


def test_trailing_week_and_baseline_bounds() -> None:
    # 周一 2026-09-28 为右端 → 上周二～本周一（起床日周窗）
    assert trailing_week_bounds(date(2026, 9, 28)) == ("2026-09-22", "2026-09-28")
    assert baseline_start("2026-09-28") == "2026-09-15"


def test_default_week_bounds_uses_local_today() -> None:
    # Asia/Shanghai 2026-09-28 12:00 → UTC 04:00
    assert default_week_bounds(
        timezone_name="Asia/Shanghai",
        now_utc=datetime(2026, 9, 28, 4, 0, tzinfo=UTC),
    ) == ("2026-09-22", "2026-09-28")


def test_align_weekly_days_fills_gaps() -> None:
    aligned = align_weekly_days(
        {
            "2026-09-28": {
                "record_date": "2026-09-28",
                "bed_time": "x",
                "wake_up_time": "y",
                "total_sleep_minutes": 450,
                "sleep_efficiency": 90,
                "rem_percent": 20,
                "deep_sleep_percent": 18,
            }
        },
        "2026-09-22",
        "2026-09-28",
    )
    assert len(aligned) == 7
    assert aligned[0] == {
        "record_date": "2026-09-22",
        "bed_time": "",
        "wake_up_time": "",
        "total_sleep_minutes": 0,
        "sleep_efficiency": 0,
        "rem_percent": 0,
        "deep_sleep_percent": 0,
    }
    assert aligned[-1]["total_sleep_minutes"] == 450


def test_resolve_report_timezone_name_fallback() -> None:
    assert resolve_report_timezone_name("") == "Asia/Shanghai"
    assert resolve_report_timezone_name("Not/AZone") == "Asia/Shanghai"
    assert resolve_report_timezone_name("UTC") == "UTC"


def test_coerce_datetime_and_format_string() -> None:
    assert coerce_datetime("2026-08-09T15:30:00+00:00") == datetime(
        2026, 8, 9, 15, 30, tzinfo=UTC
    )
    assert coerce_datetime("not-a-time") is None
    assert format_collected_at("2026-08-09T15:30:00+00:00") == (
        "2026-08-09T23:30:00+08:00"
    )
    assert format_collected_at("bad") == ""


def test_weekly_day_metric_from_record() -> None:
    metric = weekly_day_metric(
        {
            "record_date": "2026-08-10",
            "raw_data": {
                "bed_time": datetime(2026, 8, 9, 15, 30, tzinfo=UTC),
                "wake_up_time": datetime(2026, 8, 10, 0, 30, tzinfo=UTC),
                "awake_ratio": 10,
                "deep_sleep_ratio": 20,
                "light_sleep_ratio": 50,
                "rem_ratio": 20,
                "sleep_efficiency": 92,
            },
        }
    )
    assert metric["record_date"] == "2026-08-10"
    assert metric["bed_time"] == "2026-08-09T23:30:00+08:00"
    assert metric["wake_up_time"] == "2026-08-10T08:30:00+08:00"
    assert metric["total_sleep_minutes"] == 486
    assert metric["sleep_efficiency"] == 92
    assert metric["rem_percent"] == 20
    assert metric["deep_sleep_percent"] == 20


def test_weekly_day_metric_tolerates_bad_time_string() -> None:
    metric = weekly_day_metric(
        {
            "record_date": "2026-08-10",
            "raw_data": {
                "bed_time": "garbage",
                "wake_up_time": "also-bad",
                "rem_ratio": 20,
                "deep_sleep_ratio": 20,
                "sleep_efficiency": 80,
            },
        }
    )
    assert metric["bed_time"] == ""
    assert metric["wake_up_time"] == ""
    assert metric["total_sleep_minutes"] == 0
