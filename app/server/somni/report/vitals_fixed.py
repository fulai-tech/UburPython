"""GetVitals 固定 HRV / BRV 样例。"""

from __future__ import annotations

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


def fixed_hrv() -> dict[str, Any]:
    return {
        "value": 39.2,
        "personal_baseline": 35.0,
        "vs_baseline_percent": 12.0,
        "series": [dict(item) for item in _FIXED_SERIES],
    }


def fixed_brv() -> dict[str, Any]:
    return {
        "value": 56.0,
        "personal_baseline": 50.0,
        "vs_baseline_percent": 12.0,
        "series": [dict(item) for item in _FIXED_BRV_SERIES],
    }
