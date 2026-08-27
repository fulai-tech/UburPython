"""量产报告 calc 单测。"""

from __future__ import annotations

from datetime import UTC, datetime

from app.server.somni.report.calc import (
    floor_avg,
    floor_metric_stats,
    format_hhmm,
    local_day_utc_range,
    minutes_between,
    sleep_stage_parts,
    stage_minutes,
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
    assert parts["awake"]["percent"] == 10
