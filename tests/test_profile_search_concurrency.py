"""并发诊断脚本的纯函数测试。"""

from __future__ import annotations

import json

import pytest

from scripts.profile_search_concurrency import (
    diagnose,
    load_payloads,
    parse_server_timing,
    percentile,
)


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([], 95) is None
    assert percentile([10.0], 95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_percentile_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_parse_server_timing() -> None:
    assert parse_server_timing(None) == {}
    assert parse_server_timing("es;dur=12.3, embedding;dur=4.5;desc=onnx") == {
        "es": 12.3,
        "embedding": 4.5,
    }
    assert parse_server_timing('rank;dur="8.25", invalid;dur=nope') == {"rank": 8.25}


def test_load_payloads_accepts_supported_shapes(tmp_path) -> None:
    single_path = tmp_path / "single.json"
    single_path.write_text(json.dumps({"query_text": "雨声"}), encoding="utf-8")
    assert load_payloads(single_path) == [{"query_text": "雨声"}]

    list_path = tmp_path / "list.json"
    list_path.write_text(
        json.dumps([{"query_text": "雨声"}, {"query_text": "钢琴"}]),
        encoding="utf-8",
    )
    assert len(load_payloads(list_path)) == 2

    wrapped_path = tmp_path / "wrapped.json"
    wrapped_path.write_text(
        json.dumps({"requests": [{"query_text": "海浪"}]}),
        encoding="utf-8",
    )
    assert load_payloads(wrapped_path) == [{"query_text": "海浪"}]


def _summary(
    *,
    http_avg: float,
    probe_p95: float,
    es_query_avg: float | None,
    es_get_avg: float | None,
    search_queue: int = 0,
    get_queue: int = 0,
    server_timing: dict | None = None,
    profile_mget_avg: float | None = None,
) -> dict:
    return {
        "http_ms": {"avg": http_avg},
        "probe_ms": {"p95": probe_p95},
        "es": {
            "search_query_avg_ms": es_query_avg,
            "get_avg_ms": es_get_avg,
            "search_pool_queue_max": search_queue,
            "get_pool_queue_max": get_queue,
            "search_pool_rejected_delta": 0,
            "get_pool_rejected_delta": 0,
        },
        "server_timing_ms": server_timing or {},
        "profile_es_mget_calls": {"avg": profile_mget_avg},
    }


def test_diagnose_es_bottleneck() -> None:
    baseline = _summary(http_avg=100, probe_p95=10, es_query_avg=5, es_get_avg=2)
    concurrent = _summary(http_avg=300, probe_p95=12, es_query_avg=15, es_get_avg=8)
    result = diagnose(baseline, concurrent)
    assert result["verdict"].startswith("ES 侧更可疑")
    assert result["http_avg_ratio"] == 3.0


def test_diagnose_app_responsiveness_bottleneck() -> None:
    baseline = _summary(http_avg=100, probe_p95=10, es_query_avg=5, es_get_avg=2)
    concurrent = _summary(http_avg=300, probe_p95=40, es_query_avg=5, es_get_avg=2)
    result = diagnose(baseline, concurrent)
    assert result["verdict"].startswith("事件循环/应用宿主机 CPU 更可疑")


def test_diagnose_mixed_bottleneck() -> None:
    baseline = _summary(http_avg=100, probe_p95=10, es_query_avg=5, es_get_avg=2)
    concurrent = _summary(
        http_avg=300,
        probe_p95=40,
        es_query_avg=5,
        es_get_avg=2,
        search_queue=1,
    )
    result = diagnose(baseline, concurrent)
    assert result["verdict"].startswith("混合瓶颈")


def test_diagnose_prefers_server_timing_when_es_dominates() -> None:
    baseline = _summary(
        http_avg=100,
        probe_p95=10,
        es_query_avg=2,
        es_get_avg=1,
        server_timing={
            "retrieval": {"avg": 40},
            "es": {"avg": 36},
            "embedding": {"avg": 1},
            "python_other": {"avg": 3},
        },
    )
    concurrent = _summary(
        http_avg=180,
        probe_p95=30,
        es_query_avg=5,
        es_get_avg=3,
        server_timing={
            "retrieval": {"avg": 70},
            "es": {"avg": 65},
            "embedding": {"avg": 1},
            "python_other": {"avg": 4},
        },
    )

    result = diagnose(baseline, concurrent)

    assert result["verdict"].startswith("服务端阶段计时显示 ES await 主导")
    assert result["internal_es_share"] == pytest.approx(65 / 70)
    assert result["internal_es_slowdown_share"] == pytest.approx(29 / 30)


def test_diagnose_recognizes_controlled_mget_count() -> None:
    timing_baseline = {
        "retrieval": {"avg": 10},
        "es": {"avg": 7},
        "embedding": {"avg": 1},
        "python_other": {"avg": 2},
    }
    timing_concurrent = {
        "retrieval": {"avg": 20},
        "es": {"avg": 17},
        "embedding": {"avg": 1},
        "python_other": {"avg": 2},
    }
    baseline = _summary(
        http_avg=12,
        probe_p95=4,
        es_query_avg=1,
        es_get_avg=1,
        server_timing=timing_baseline,
        profile_mget_avg=1,
    )
    concurrent = _summary(
        http_avg=24,
        probe_p95=8,
        es_query_avg=2,
        es_get_avg=2,
        server_timing=timing_concurrent,
        profile_mget_avg=1,
    )

    result = diagnose(baseline, concurrent)

    assert result["verdict"].startswith("N+1 mget 已受控")
    assert result["profile_es_mget_calls_avg"] == 1
