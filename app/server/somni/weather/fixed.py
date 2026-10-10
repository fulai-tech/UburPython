"""天气固定样例（GetWeather / GetDailyAggregate 共用）。"""

from __future__ import annotations

from typing import Any


def fixed_weather(record_date: str = "2026-09-24") -> dict[str, Any]:
    """本期不查外部源；结构固定，date 可跟入参 record_date。"""
    day = (record_date or "").strip() or "2026-09-24"
    return {
        "date": day,
        "temperature": {"min": 15, "max": 29},
        "condition_code": "partly_cloudy_rain",
        "sunrise_at": f"{day}T07:00:00+08:00",
        "sunset_at": f"{day}T20:00:00+08:00",
        "uv_index": {"level": "low"},
        "humidity_percent": 58,
        "aqi": "hazardous",
    }
