"""GetVitals 固定 HRV / BRV 样例（数值均向下取整）。"""

from __future__ import annotations

import math
from typing import Any

_FIXED_SERIES = [
    {"collected_at": "2026-09-23T23:00:00+08:00", "value": 36.0},
    {"collected_at": "2026-09-24T01:00:00+08:00", "value": 38.0},
    {"collected_at": "2026-09-24T03:00:00+08:00", "value": 39.0},
    {"collected_at": "2026-09-24T05:00:00+08:00", "value": 42.0},
    {"collected_at": "2026-09-24T07:00:00+08:00", "value": 41.0},
]

_FIXED_BRV_SERIES = [
    {"collected_at": "2026-09-23T23:00:00+08:00", "value": 50.0},
    {"collected_at": "2026-09-24T01:00:00+08:00", "value": 54.0},
    {"collected_at": "2026-09-24T03:00:00+08:00", "value": 56.0},
    {"collected_at": "2026-09-24T05:00:00+08:00", "value": 60.0},
    {"collected_at": "2026-09-24T07:00:00+08:00", "value": 60.0},
]


def _floor_num(value: float) -> float:
    return float(math.floor(value))


def _floor_series(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "collected_at": str(item["collected_at"]),
            "value": _floor_num(float(item["value"])),
        }
        for item in items
    ]


def fixed_hrv() -> dict[str, Any]:
    return {
        "value": _floor_num(39.2),
        "personal_baseline": _floor_num(35.0),
        "vs_baseline_percent": _floor_num(12.0),
        "series": _floor_series(_FIXED_SERIES),
    }


def fixed_brv() -> dict[str, Any]:
    return {
        "value": _floor_num(56.0),
        "personal_baseline": _floor_num(50.0),
        "vs_baseline_percent": _floor_num(12.0),
        "series": _floor_series(_FIXED_BRV_SERIES),
    }
