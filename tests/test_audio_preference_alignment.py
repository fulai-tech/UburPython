"""Questionnaire preferences resolve before any retrieval or rule calculation."""

import pytest

from app.core import audio_tags as tags

RAIN = {
    "steady_light_rain",
    "steady_rain",
    "heavy_rain",
    "surface_rain",
    "scene_rain",
    "rainforest_light_rain",
    "storm_ambience",
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


def test_unknown_explicit_dislike_closes_sound_instead_of_dropping_constraint():
    resolved = tags.resolve_audio_preferences(preferred=["溪流"], disliked=["未知但我不喜欢的声音"])
    assert resolved["preferred_codes"] == ["stream"]
    assert resolved["unresolved_disliked"] == ["未知但我不喜欢的声音"]
    assert resolved["sound_disabled"]


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
    assert resolved["sound_disabled"]


def test_legacy_rain_input_is_aligned_before_strict_request_validation():
    from app.schemas.audio import SearchAudioRequest

    request = SearchAudioRequest(content_tags=["rain sounds"], disliked_tags=["鸟鸣"])
    assert set(request.content_tags) == RAIN
    assert set(request.disliked_tags) == {"bird_calls", "dawn_chorus"}
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
