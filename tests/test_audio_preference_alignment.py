"""Questionnaire preferences resolve before any retrieval or rule calculation."""

import pytest

from app.core import audio_tags as tags

RAIN = {
    "steady_light_rain",
    "steady_rain",
    "heavy_rain",
    "rain_on_window",
    "thunder",
}


def test_rain_dislike_resolves_equally_in_both_languages_without_banning_all_nature():
    zh = tags.resolve_audio_preferences(disliked=["雨声"])
    en = tags.resolve_audio_preferences(disliked=["rain sounds"])
    assert set(zh["disliked_codes"]) == RAIN
    assert zh["disliked_codes"] == en["disliked_codes"]
    assert "stream" not in zh["disliked_codes"]
    assert "natural_sound" not in zh["disliked_codes"]
    assert not zh["sound_disabled"]


def test_non_dictionary_english_is_not_a_valid_search_tag():
    with pytest.raises(ValueError, match="unknown_audio_tag"):
        tags.validated_audio_tags(["rain_sound"])


def test_parent_dislike_covers_children_but_not_other_categories():
    resolved = tags.resolve_audio_preferences(disliked=["噪音"])
    assert {"color_noise", "white_noise", "pink_noise", "brown_noise"} <= set(
        resolved["disliked_codes"]
    )
    assert "stream" not in resolved["disliked_codes"]


def test_unknown_free_text_is_diagnostic_without_disabling_sound():
    resolved = tags.resolve_audio_preferences(preferred=["溪流"], disliked=["未知但我不喜欢的声音"])
    assert resolved["preferred_codes"] == ["stream"]
    assert resolved["unresolved_disliked"] == ["未知但我不喜欢的声音"]
    assert not resolved["sound_disabled"]


def test_none_is_only_no_exclusion_in_a_dislike_answer():
    resolved = tags.resolve_audio_preferences(disliked=["无", "none", "no dislikes"])
    assert resolved["disliked_codes"] == []
    assert not resolved["sound_disabled"]
    assert tags.canonical_tag("无") == "none"


def test_silence_is_a_control_choice_and_not_an_audio_category():
    resolved = tags.resolve_audio_preferences(preferred=["安静无声"])
    assert resolved["sound_disabled"]
    assert resolved["preferred_codes"] == []


def test_sentence_dislike_uses_bounded_mapping_not_english_slug_invention():
    resolved = tags.resolve_audio_preferences(disliked=["不喜欢雨声", "avoid rain sounds"])
    assert set(resolved["disliked_codes"]) == RAIN
    assert not resolved["unresolved_disliked"]


def test_types_do_not_allow_a_stage_to_become_a_content_preference():
    resolved = tags.resolve_audio_preferences(disliked=["soothe"])
    assert resolved["unresolved_disliked"] == ["soothe"]
    assert not resolved["sound_disabled"]


def test_legacy_rain_input_is_aligned_before_strict_request_validation():
    from app.schemas.audio import SearchAudioRequest

    request = SearchAudioRequest(content_tags=["rain sounds"], disliked_tags=["鸟鸣"])
    assert set(request.content_tags) == RAIN
    assert set(request.disliked_tags) == {"birdsong", "dawn_chorus"}
    with pytest.raises(ValueError):
        SearchAudioRequest(content_tag_codes=["rain_sound"])


def test_search_cache_identity_ignores_tag_order_and_includes_mapping_revision(monkeypatch):
    import app.cache.audio_search_cache as cache
    from app.schemas.audio import SearchAudioRequest

    a = SearchAudioRequest(content_tags=["stream", "steady_rain"])
    b = SearchAudioRequest(content_tags=["steady_rain", "stream", "stream"])
    assert cache.build_audio_search_cache_key(a) == cache.build_audio_search_cache_key(b)
    before = cache.build_audio_search_cache_key(a)
    monkeypatch.setattr(cache, "audio_tag_contract_version", lambda: "next-version")
    assert cache.build_audio_search_cache_key(a) != before


@pytest.mark.asyncio
async def test_high_vector_similarity_cannot_turn_a_stream_into_banned_rain():
    from unittest.mock import MagicMock

    from app.core.config import Settings
    from app.es.search import EsSearch
    from app.services.retrieval import RetrievalService, ScoredCandidate

    source = {"content_form_tags": [{"code": "stream", "tag_id": "stream-id", "name": "溪流"}]}
    candidate = ScoredCandidate(source=source, tags=EsSearch.parse_tags(source))
    service = RetrievalService(MagicMock(), MagicMock(), Settings())
    assert await service._apply_dislike_filter(
        [candidate], ["steady_rain"], {"stream-id": [1.0, 0.0]}, dislike_vectors=[[1.0, 0.0]]
    ) == [candidate]
    assert (
        service._dislike_penalty(
            candidate.tags, ["steady_rain"], [[1.0, 0.0]], {"stream-id": [1.0, 0.0]}
        )
        < 1.0
    )


def test_broken_dictionary_hierarchy_is_rejected_before_serving():
    with pytest.raises(ValueError, match="parent"):
        tags.validate_audio_tag_contract(
            [
                {
                    "id": "one",
                    "code": "stream",
                    "type": "content_form",
                    "status": "启用",
                    "parent_tag_id": "missing",
                }
            ],
            {"groups": []},
        )


@pytest.mark.asyncio
async def test_negative_rain_query_aligns_all_codes_before_candidate_recall():
    from unittest.mock import AsyncMock, MagicMock

    from app.core.config import Settings
    from app.services.retrieval import RetrievalService

    service = RetrievalService(MagicMock(), MagicMock(), Settings())
    service._encode_one = AsyncMock(return_value=[1.0, 0.0])
    result = await service._extract_query_tags("不要雨声", [1.0, 0.0], [])
    assert set(result.disliked_tags) == RAIN
    assert not result.unresolved_dislike


@pytest.mark.parametrize(
    "voice_value,excluded", [("none", False), ("continuous", True), (None, True)]
)
def test_authoritative_material_voice_check_does_not_confuse_field_name_with_value(
    voice_value, excluded
):
    material = {"audio_engineering_tags": [{"code": "voice", "value": {"code": voice_value}}]}
    assert tags.material_is_excluded(material, ["voice"]) is excluded


def test_preference_alias_cannot_override_the_dictionary_name():
    with pytest.raises(ValueError, match="alias"):
        tags.validate_audio_tag_contract(
            tags.dictionary_records(),
            {"groups": [{"aliases": ["均匀小雨"], "codes": ["heavy_rain"]}]},
        )


@pytest.mark.parametrize(
    "term",
    [
        "钢琴",
        "钢琴声",
        "钢琴曲",
        "piano",
        "piano music",
        "PIANO SOUNDS",
        "我不想听钢琴",
        "I don't like piano",
    ],
)
def test_common_piano_words_resolve_to_existing_code_without_disabling_sound(term):
    resolved = tags.resolve_audio_preferences(preferred=["自然声"], disliked=[term, "库外类别"])
    assert resolved["disliked_codes"] == ["slow_piano"]
    assert not resolved["sound_disabled"]
    assert resolved["unresolved_disliked"] == ["库外类别"]
    assert tags.canonical_disliked_tags([term, "库外类别"]) == ["slow_piano"]


@pytest.mark.parametrize(
    "term,expected",
    [
        ("鸟叫", {"birdsong", "dawn_chorus"}),
        ("海浪", {"gentle_ocean_wave"}),
        ("溪水声", {"stream"}),
        ("风扇声", {"fan_sound"}),
        ("白噪声", {"white_noise"}),
        ("粉红噪音", {"pink_noise"}),
        ("棕色噪声", {"brown_noise"}),
        ("竖琴", set()),
        ("手碟", set()),
        ("颂钵", set()),
    ],
)
def test_reviewed_terms_follow_current_catalog_without_guessing(term, expected):
    resolved = tags.resolve_audio_preferences(disliked=[term])
    assert set(resolved["disliked_codes"]) == expected
    assert not resolved["sound_disabled"]


@pytest.mark.parametrize(
    "term",
    ["吉他", "摇滚", "重金属", "不存在的猫叫声种类", "unknown_ocean_type", "clock_tick_unknown"],
)
def test_out_of_catalog_words_are_ignored_without_substring_guessing(term):
    assert tags.canonical_disliked_tags([term]) == []
    assert not tags.resolve_audio_preferences(disliked=[term])["sound_disabled"]


def test_mixed_separators_keep_all_known_dislikes_and_ignore_unknown():
    resolved = tags.resolve_audio_preferences(disliked=["不要钢琴/鸟叫；吉他\n雨声"])
    assert {"slow_piano", "birdsong", "dawn_chorus"} | RAIN == set(resolved["disliked_codes"])
    assert not resolved["sound_disabled"]


def test_unknown_words_do_not_override_explicit_silence_or_machine_tag_validation():
    assert tags.resolve_audio_preferences(preferred=["安静无声"], disliked=["库外类别"])[
        "sound_disabled"
    ]
    with pytest.raises(ValueError, match="unknown_audio_tag"):
        tags.validated_audio_tags(["not_in_dictionary"])


def test_valid_body_focus_blocks_survive_unknown_free_text():
    resolved = tags.resolve_audio_preferences(
        disliked=["库外类别"],
        blocked=["body_scan", "breathing_guidance", "progressive_muscle_relaxation"],
    )
    assert set(resolved["disliked_codes"]) == {
        "body_scan",
        "breathing_guidance",
        "progressive_muscle_relaxation",
    }
    assert not resolved["sound_disabled"]


@pytest.mark.parametrize(
    "term", ["我不喜欢听钢琴", "不要放钢琴", "讨厌钢琴的声音", "don't like piano music"]
)
def test_conventional_negative_phrases_keep_reviewed_piano_exclusion(term):
    assert tags.canonical_disliked_tags([term]) == ["slow_piano"]


def test_dictionary_identity_is_not_broadened_by_a_colloquial_alias():
    assert tags.canonical_disliked_tags(["birdsong"]) == ["birdsong"]
    assert tags.canonical_disliked_tags(["Bird Calls"]) == ["birdsong", "dawn_chorus"]


def test_unknown_legacy_preferences_share_cache_identity_after_ignoring_text():
    from app.cache.audio_search_cache import build_audio_search_cache_key
    from app.schemas.audio import SearchAudioRequest

    a = SearchAudioRequest(content_tags=["钢琴", "不存在的音乐"], disliked_tags=["库外类别"])
    b = SearchAudioRequest(content_tags=["slow_piano"])
    assert a.content_tags == ["slow_piano"]
    assert a.disliked_tags == []
    assert build_audio_search_cache_key(a) == build_audio_search_cache_key(b)


def test_no_dislike_marker_in_a_list_is_not_an_engineering_exclusion():
    resolved = tags.resolve_audio_preferences(disliked=["无、钢琴", "none and unknown category"])
    assert resolved["disliked_codes"] == ["slow_piano"]
    assert not resolved["sound_disabled"]


def test_explicit_silence_inside_a_preference_list_is_preserved():
    assert tags.resolve_audio_preferences(preferred=["未知类别，安静无声"])["sound_disabled"]


def test_negative_text_cannot_become_a_positive_preference():
    resolved = tags.resolve_audio_preferences(preferred=["不要钢琴"])
    assert resolved["preferred_codes"] == []


def test_full_width_list_separators_keep_known_exclusions():
    assert set(tags.canonical_disliked_tags(["ＰＩＡＮＯ／鸟叫"])) == {
        "slow_piano",
        "birdsong",
        "dawn_chorus",
    }
