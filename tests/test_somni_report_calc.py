"""量产报告 calc 单测。"""

from __future__ import annotations

from datetime import UTC, date, datetime

from app.server.somni.report.calc import (
    STAGE_AWAKE,
    STAGE_DEEP,
    STAGE_LIGHT,
    STAGE_REM,
    align_weekly_days,
    allocate_percentages,
    baseline_start,
    coerce_datetime,
    default_week_bounds,
    floor_avg,
    floor_metric_stats,
    format_collected_at,
    format_hhmm,
    format_local_ms_hhmm,
    idf_from_segments,
    local_day_utc_range,
    minutes_between,
    on_bed_bounds,
    on_bed_duration_ms,
    resolve_report_timezone_name,
    session_timezone_name,
    sleep_quality_from_session,
    sleep_stage_parts,
    sleep_structure_from_segments,
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


def test_format_local_ms_hhmm_converts_to_timezone() -> None:
    ms = int(datetime(2026, 11, 10, 15, 59, tzinfo=UTC).timestamp() * 1000)
    assert format_local_ms_hhmm(ms, "UTC") == "15:59"
    assert format_local_ms_hhmm(ms, "Asia/Shanghai") == "23:59"
    assert format_local_ms_hhmm(ms, "") == "23:59"
    assert format_local_ms_hhmm(0, "Asia/Shanghai") == ""


def test_session_timezone_name_prefers_payload() -> None:
    assert session_timezone_name({"timezone": "UTC"}, {"timezone": "Asia/Shanghai"}) == (
        "UTC"
    )
    assert session_timezone_name({}, {"timezone": "America/Los_Angeles"}) == (
        "America/Los_Angeles"
    )
    assert session_timezone_name({}, {}) == "Asia/Shanghai"


def test_idf_last_end_padded_to_on_bed_to() -> None:
    bed_from = int(datetime(2026, 11, 4, 22, 45, tzinfo=UTC).timestamp() * 1000)
    bed_to = int(datetime(2026, 11, 5, 7, 15, tzinfo=UTC).timestamp() * 1000)
    on_bed = [{"from": bed_from, "to": bed_to, "duration_ms": bed_to - bed_from}]
    segments = [
        {
            "stage": 2,
            "stage_name": "LIGHT",
            "duration_ms": 60 * 60_000,
            "start_ts_ms": bed_from,
        },
        {
            "stage": 1,
            "stage_name": "AWAKE",
            "duration_ms": 14 * 60_000,
            "start_ts_ms": bed_from + 8 * 3600_000,
        },
    ]
    idf = idf_from_segments(segments, on_bed, timezone_name="UTC")
    assert idf[-1]["start"] == "06:45"
    assert idf[-1]["end"] == "07:15"
    # 不传 on_bed 时仍按 duration
    raw = idf_from_segments(segments, timezone_name="UTC")
    assert raw[-1]["end"] == "06:59"
    sh = idf_from_segments(segments, on_bed, timezone_name="Asia/Shanghai")
    assert sh[0]["start"] == "06:45"
    assert sh[-1]["start"] == "14:45"
    assert sh[-1]["end"] == "15:15"


def test_on_bed_bounds_uses_first_from_and_last_to() -> None:
    on_bed = [
        {"from": 1_000_000, "to": 1_600_000, "duration_ms": 600_000},
        {"from": 2_000_000, "to": 2_900_000, "duration_ms": 900_000},
    ]
    assert on_bed_bounds(on_bed) == (1_000_000, 2_900_000)
    assert on_bed_duration_ms(on_bed) == 1_500_000


def test_multi_on_bed_structure_sums_raw_ms_then_floors() -> None:
    # 两段在床：10.5min + 20.4min = 30.9 → TIB 30；离床间隙不算
    base = 10_000_000
    on_bed = [
        {"from": base, "to": base + 630_000, "duration_ms": 630_000},
        {"from": base + 1_000_000, "to": base + 2_224_000, "duration_ms": 1_224_000},
    ]
    # 每段不足整分的毫秒：先按阶段汇总再 floor，避免按段 floor 丢时长
    segments = [
        {
            "stage": STAGE_AWAKE,
            "stage_name": "AWAKE",
            "duration_ms": 90_000,
            "start_ts_ms": base,
        },
        {
            "stage": STAGE_LIGHT,
            "stage_name": "LIGHT",
            "duration_ms": 590_000,
            "start_ts_ms": base + 90_000,
        },
        {
            "stage": STAGE_DEEP,
            "stage_name": "DEEP",
            "duration_ms": 610_000,
            "start_ts_ms": base + 1_000_000,
        },
        {
            "stage": STAGE_REM,
            "stage_name": "REM",
            "duration_ms": 500_000,
            "start_ts_ms": base + 1_610_000,
        },
        {
            "stage": STAGE_AWAKE,
            "stage_name": "AWAKE",
            "duration_ms": 120_000,
            "start_ts_ms": base + 2_110_000,
        },
    ]
    structure = sleep_structure_from_segments(on_bed, segments)
    assert structure["tib_minutes"] == 30
    assert structure["awake"]["minutes"] == 3  # 210000ms
    assert structure["light_sleep"]["minutes"] == 9  # 590000ms
    assert structure["deep_sleep"]["minutes"] == 10  # 610000ms
    assert structure["rem_sleep"]["minutes"] == 8  # 500000ms
    assert structure["total_minutes"] == 28  # 1700000ms，而非 9+10+8 的展示加总
    # 按原始 ms 分配：210/500/590/610 → 11/26/31/32
    assert structure["awake"]["percent"] == 11
    assert structure["rem_sleep"]["percent"] == 26
    assert structure["light_sleep"]["percent"] == 31
    assert structure["deep_sleep"]["percent"] == 32
    assert (
        structure["awake"]["percent"]
        + structure["rem_sleep"]["percent"]
        + structure["light_sleep"]["percent"]
        + structure["deep_sleep"]["percent"]
        == 100
    )


def test_allocate_percentages_sums_to_100() -> None:
    assert allocate_percentages([64, 99, 266, 70]) == [13, 20, 53, 14]
    assert sum(allocate_percentages([64, 99, 266, 70])) == 100
    assert allocate_percentages([0, 0, 0, 0]) == [0, 0, 0, 0]


def test_sleep_structure_percent_keeps_sub_minute_stages() -> None:
    """不足 1 分钟的阶段 minutes=0，但 percent 仍按 ms 保留份额。"""
    bed_from = 1_000_000
    on_bed = [{"from": bed_from, "to": bed_from + 360_000, "duration_ms": 360_000}]
    cursor = bed_from
    segments = []
    for stage, name, duration_ms in (
        (STAGE_AWAKE, "AWAKE", 59_000),
        (STAGE_REM, "REM", 59_000),
        (STAGE_LIGHT, "LIGHT", 59_000),
        (STAGE_DEEP, "DEEP", 183_000),
    ):
        segments.append(
            {
                "stage": stage,
                "stage_name": name,
                "duration_ms": duration_ms,
                "start_ts_ms": cursor,
            }
        )
        cursor += duration_ms
    structure = sleep_structure_from_segments(on_bed, segments)
    assert structure["awake"]["minutes"] == 0
    assert structure["rem_sleep"]["minutes"] == 0
    assert structure["light_sleep"]["minutes"] == 0
    assert structure["deep_sleep"]["minutes"] == 3
    assert structure["awake"]["percent"] == 17
    assert structure["rem_sleep"]["percent"] == 16
    assert structure["light_sleep"]["percent"] == 16
    assert structure["deep_sleep"]["percent"] == 51
    assert (
        structure["awake"]["percent"]
        + structure["rem_sleep"]["percent"]
        + structure["light_sleep"]["percent"]
        + structure["deep_sleep"]["percent"]
        == 100
    )


def test_sleep_structure_percent_uses_stage_minutes_not_tib() -> None:
    # TIB=454，四阶段分钟和=499；占比按阶段时长切到 100%，不能按 TIB floor
    bed_from = 1_000_000
    tib_ms = 454 * 60_000
    on_bed = [{"from": bed_from, "to": bed_from + tib_ms, "duration_ms": tib_ms}]
    cursor = bed_from
    segments = []
    for stage, name, minutes in (
        (STAGE_AWAKE, "AWAKE", 64),
        (STAGE_REM, "REM", 99),
        (STAGE_LIGHT, "LIGHT", 266),
        (STAGE_DEEP, "DEEP", 70),
    ):
        duration = minutes * 60_000
        segments.append(
            {
                "stage": stage,
                "stage_name": name,
                "duration_ms": duration,
                "start_ts_ms": cursor,
            }
        )
        cursor += duration
    structure = sleep_structure_from_segments(on_bed, segments)
    assert structure["tib_minutes"] == 454
    assert structure["awake"] == {"minutes": 64, "percent": 13}
    assert structure["rem_sleep"] == {"minutes": 99, "percent": 20}
    assert structure["light_sleep"] == {"minutes": 266, "percent": 53}
    assert structure["deep_sleep"] == {"minutes": 70, "percent": 14}


def test_sleep_quality_wake_to_leave_bed_uses_last_on_bed_to_minus_last_awake_start() -> None:
    on_bed = [
        {"from": 1_000_000, "to": 1_600_000, "duration_ms": 600_000},
        {"from": 2_000_000, "to": 3_200_000, "duration_ms": 1_200_000},
    ]
    segments = [
        {
            "stage": STAGE_AWAKE,
            "stage_name": "AWAKE",
            "duration_ms": 300_000,
            "start_ts_ms": 1_000_000,
        },
        {
            "stage": STAGE_LIGHT,
            "stage_name": "LIGHT",
            "duration_ms": 900_000,
            "start_ts_ms": 1_300_000,
        },
        {
            "stage": STAGE_AWAKE,
            "stage_name": "AWAKE",
            "duration_ms": 120_000,
            "start_ts_ms": 2_500_000,
        },
        {
            "stage": STAGE_LIGHT,
            "stage_name": "LIGHT",
            "duration_ms": 480_000,
            "start_ts_ms": 2_620_000,
        },
    ]
    quality = sleep_quality_from_session(on_bed, segments, timezone_name="UTC")
    assert quality["time_in_bed_minutes"] == 30
    assert quality["sleep_onset_latency_minutes"] == 5  # 300000ms
    # 醒后到离床 = last on_bed.to(3200000) - last awake start(2500000) = 700000ms → 11
    assert quality["wake_to_leave_bed_minutes"] == 11
    assert quality["bedtime"] == "00:16"
    assert quality["wake_up_time"] == "00:53"
    assert quality["sleep_efficiency"] == 76  # TST 23 / TIB 30
    sh = sleep_quality_from_session(on_bed, segments, timezone_name="Asia/Shanghai")
    assert sh["bedtime"] == "08:16"
    assert sh["wake_up_time"] == "08:53"
