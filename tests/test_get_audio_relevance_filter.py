"""GetAudio 相关过滤：门槛 / 断崖 / 颜色噪音互斥。"""

from app.core.config import Settings
from app.server.somni.audio.relevance_filter import (
    apply_lane_relevance_filters,
    detect_color_noise_intents,
    diversify_multi_color_noise,
    filter_conflicting_color_noise,
    filter_docs_by_min_score,
    filter_docs_by_score_cliff,
)


def test_resolve_cliff_ratio_by_language() -> None:
    s = Settings(
        _env_file=None,
        get_audio_score_cliff_ratio=0.97,
        get_audio_score_cliff_ratio_zh=0.93,
        get_audio_score_cliff_ratio_en=0.95,
    )
    assert s.resolve_get_audio_score_cliff_ratio("zh") == 0.93
    assert s.resolve_get_audio_score_cliff_ratio("zh-CN") == 0.93
    assert s.resolve_get_audio_score_cliff_ratio("en") == 0.95
    assert s.resolve_get_audio_score_cliff_ratio("en-US") == 0.95
    assert s.resolve_get_audio_score_cliff_ratio("ja") == 0.97
    s2 = Settings(
        _env_file=None,
        get_audio_score_cliff_ratio=0.97,
        get_audio_score_cliff_ratio_zh=None,
        get_audio_score_cliff_ratio_en=None,
    )
    assert s2.resolve_get_audio_score_cliff_ratio("zh") == 0.97
    assert s2.resolve_get_audio_score_cliff_ratio("en") == 0.97


def test_min_score_drops_weak_hits() -> None:
    docs = [
        {"_id": "a", "_knn_score": 0.90},
        {"_id": "b", "_knn_score": 0.84},
        {"_id": "c", "_knn_score": 0.70},
    ]
    kept = filter_docs_by_min_score(docs, score_field="_knn_score", min_score=0.85)
    assert [d["_id"] for d in kept] == ["a"]


def test_score_cliff_keeps_near_top_only() -> None:
    docs = [
        {"_id": "a", "_knn_score": 0.90},
        {"_id": "b", "_knn_score": 0.88},
        {"_id": "c", "_knn_score": 0.80},
    ]
    kept = filter_docs_by_score_cliff(
        docs, score_field="_knn_score", cliff_ratio=0.97
    )
    assert [d["_id"] for d in kept] == ["a", "b"]


def test_apply_lane_skips_when_no_scores() -> None:
    docs = [{"_id": "a"}, {"_id": "b"}]
    assert apply_lane_relevance_filters(
        docs, score_field="_knn_score", min_score=0.85, cliff_ratio=0.97
    ) == docs


def test_detect_white_noise_intent() -> None:
    assert detect_color_noise_intents("white noise") == ["白噪音"]
    assert detect_color_noise_intents("What white noise is good for deep sleep?") == [
        "白噪音"
    ]
    assert detect_color_noise_intents("白噪音") == ["白噪音"]


def test_filter_drops_sibling_color_noise() -> None:
    docs = [
        {"_id": "w", "audio_name": "Rhythmic White Noise"},
        {"_id": "b", "audio_name": "Stop Daydreaming Smooth Brown Noise (Pilot)"},
        {"_id": "p", "audio_name": "Pink Noise (Pilot)"},
        {"_id": "r", "audio_name": "Cozy Apartment Rain"},
    ]
    kept = filter_conflicting_color_noise(docs, query_text="white noise")
    assert [d["_id"] for d in kept] == ["w", "r"]


def test_filter_uses_title_over_mismatched_tags() -> None:
    """标题粉噪、标签误标棕噪：查棕噪应剔除。"""
    docs = [
        {
            "_id": "p",
            "audio_name": "Soft Pink Noise (Pilot)",
            "content_form_tags": [
                {"name": "Brown Noise", "name_en": "Brown Noise", "code": "brown_noise"}
            ],
            "short_description": "deep brown noise hum",
        },
        {
            "_id": "b",
            "audio_name": "Sleep Brown Noise (Pilot Test)",
            "content_form_tags": [
                {"name": "Brown Noise", "code": "brown_noise"}
            ],
        },
    ]
    kept = filter_conflicting_color_noise(docs, query_text="brown noise")
    assert [d["_id"] for d in kept] == ["b"]


def test_filter_drops_white_title_when_desc_mentions_brown() -> None:
    docs = [
        {
            "_id": "w",
            "audio_name": "Sleep Relaxation White Noise (Pre-trial)",
            "short_description": "Stable hum with rich brown noise",
        }
    ]
    kept = filter_conflicting_color_noise(docs, query_text="brown noise")
    assert kept == []


def test_filter_keeps_when_query_not_color_noise() -> None:
    docs = [
        {"_id": "b", "audio_name": "Brown Noise Sound Healing Journey"},
        {"_id": "r", "audio_name": "Rain"},
    ]
    assert filter_conflicting_color_noise(docs, query_text="rain") == docs


def test_diversify_multi_color_keeps_each_intent() -> None:
    docs = [
        {"_id": "p", "audio_name": "粉噪音"},
        {"_id": "x", "audio_name": "雨声"},
        {"_id": "w", "audio_name": "白噪音"},
        {"_id": "b", "audio_name": "棕噪音"},
    ]
    kept = diversify_multi_color_noise(
        docs, query_text="白噪音和粉噪音", per_color=1
    )
    assert [d["_id"] for d in kept[:2]] == ["w", "p"]
    assert {d["_id"] for d in kept} == {"w", "p", "x"}
