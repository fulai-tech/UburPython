"""GetAudio 相关过滤：门槛 / 断崖 / 颜色噪音互斥。"""

from app.core.config import Settings
from app.server.somni.audio.relevance_filter import (
    apply_lane_relevance_filters,
    detect_color_noise_intents,
    diversify_multi_color_noise,
    doc_matches_query_lexically,
    filter_conflicting_color_noise,
    filter_docs_by_min_score,
    filter_docs_by_query_lexical_match,
    filter_docs_by_score_cliff,
    is_bm25_query_blacklisted,
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


def test_resolve_lane_cliff_ratio_overrides_shared() -> None:
    s = Settings(
        _env_file=None,
        get_audio_score_cliff_ratio=0.97,
        get_audio_bm25_score_cliff_ratio=0.85,
        get_audio_knn_score_cliff_ratio=0.90,
    )
    assert s.resolve_get_audio_bm25_score_cliff_ratio("zh") == 0.85
    assert s.resolve_get_audio_knn_score_cliff_ratio("zh") == 0.90
    s2 = Settings(
        _env_file=None,
        get_audio_score_cliff_ratio=0.97,
        get_audio_bm25_score_cliff_ratio=None,
        get_audio_knn_score_cliff_ratio=None,
    )
    assert s2.resolve_get_audio_bm25_score_cliff_ratio("zh") == 0.97
    assert s2.resolve_get_audio_knn_score_cliff_ratio("en") == 0.97


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


def test_multi_color_lane_cliff_keeps_both_colors() -> None:
    """全局断崖会砍掉粉；按色断崖后白+粉保留，雨声等非意图色丢弃。"""
    docs = [
        {"_id": "w", "audio_name": "白噪音", "_bm25_score": 10.0},
        {"_id": "p", "audio_name": "粉噪音", "_bm25_score": 8.9},
        {"_id": "x", "audio_name": "雨声", "_bm25_score": 5.0},
    ]
    global_cliff = apply_lane_relevance_filters(
        docs,
        score_field="_bm25_score",
        min_score=0.0,
        cliff_ratio=0.90,
    )
    assert [d["_id"] for d in global_cliff] == ["w"]

    per_color = apply_lane_relevance_filters(
        docs,
        score_field="_bm25_score",
        min_score=0.0,
        cliff_ratio=0.90,
        color_intents=["白噪音", "粉噪音"],
    )
    assert {d["_id"] for d in per_color} == {"w", "p"}
    after_div = diversify_multi_color_noise(
        per_color, query_text="白噪音和粉噪音", per_color=1
    )
    assert {d["_id"] for d in after_div} == {"w", "p"}


def test_bm25_query_blacklist_blocks_particles() -> None:
    assert is_bm25_query_blacklisted("了")
    assert is_bm25_query_blacklisted("我")
    assert is_bm25_query_blacklisted("和")
    assert is_bm25_query_blacklisted("The")
    assert is_bm25_query_blacklisted("no")
    assert is_bm25_query_blacklisted("car")
    assert is_bm25_query_blacklisted("ok")  # ≤2 纯字母
    assert is_bm25_query_blacklisted("zz")  # 极短英文规则
    assert not is_bm25_query_blacklisted("雨声")
    assert not is_bm25_query_blacklisted("人声")
    assert not is_bm25_query_blacklisted("噪声")
    assert not is_bm25_query_blacklisted("雨")
    assert not is_bm25_query_blacklisted("助眠")
    assert not is_bm25_query_blacklisted("rain")
    assert not is_bm25_query_blacklisted("wind")
    # 空配置 / 空集合必须回退内置默认，不能静默失效
    assert is_bm25_query_blacklisted("了", blacklist=frozenset())
    assert is_bm25_query_blacklisted(
        "了",
        blacklist=Settings(
            _env_file=None, get_audio_bm25_query_blacklist=""
        ).resolve_get_audio_bm25_query_blacklist(),
    )


def test_lexical_match_keeps_title_or_tag_overlap() -> None:
    rain = {
        "_id": "1",
        "audio_name": "轻柔微雨",
        "content_form_tags": [{"name": "均匀小雨", "code": "steady_light_rain"}],
    }
    piano = {
        "_id": "2",
        "audio_name": "静待相逢",
        "content_form_tags": [{"name": "轻音乐", "code": "light_music"}],
    }
    voice = {
        "_id": "3",
        "audio_name": "正念冥想",
        "content_form_tags": [{"name": "人声引导", "code": "voice_guidance"}],
    }
    assert doc_matches_query_lexically(rain, "雨声")
    assert not doc_matches_query_lexically(piano, "钢琴")
    assert doc_matches_query_lexically(voice, "人声")
    kept = filter_docs_by_query_lexical_match(
        [rain, piano, voice], query_text="钢琴"
    )
    assert kept == []


def test_lexical_match_is_language_pure() -> None:
    rain = {
        "_id": "1",
        "audio_name": "雨落林间",
        "audio_name_en": "Rain in Woods",
        "content_form_tags": [
            {
                "name": "场景雨声",
                "code": "scene_rain",
                "en_name": "Rain in Different Settings",
            }
        ],
    }
    assert not doc_matches_query_lexically(rain, "rain", language="zh")
    assert doc_matches_query_lexically(rain, "rain", language="en")
    assert doc_matches_query_lexically(rain, "雨", language="zh")
    assert not doc_matches_query_lexically(rain, "雨", language="en")
