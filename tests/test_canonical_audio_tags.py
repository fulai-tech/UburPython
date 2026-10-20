"""English tag codes govern retrieval and cache identity in every locale."""

import pytest

from app.cache.audio_search_cache import build_audio_search_cache_key
from app.cache.sleep_stage_cache import SLEEP_STAGES, build_sleep_stage_index_key
from app.es.search import EsSearch, _sleep_stage_filter
from app.schemas.audio import SearchAudioRequest


def test_equivalent_language_tags_share_search_cache_key():
    zh = SearchAudioRequest(
        sleep_stage_tags=["入睡"], content_tags=["均匀小雨"], disliked_tags=["呼吸引导"], top_k=5
    )
    en = SearchAudioRequest(
        sleep_stage_tags=["soothe"],
        content_tags=["steady_light_rain"],
        disliked_tags=["breathing_guidance"],
        top_k=5,
    )
    assert zh.model_dump() == en.model_dump()
    assert build_audio_search_cache_key(zh) == build_audio_search_cache_key(en)


def test_stage_index_key_and_warm_names_are_english():
    assert build_sleep_stage_index_key("放松") == build_sleep_stage_index_key("unwind")
    assert build_sleep_stage_index_key("W-S3").endswith(":wake")
    assert SLEEP_STAGES == ("unwind", "soothe", "guard", "wake")


def test_material_matching_prefers_code_and_accepts_legacy_display_aliases():
    tags = EsSearch.parse_tags(
        {"content_form_tags": [{"tag_id": "id1", "code": "breathing_guidance", "name": "呼吸引导"}]}
    )
    assert tags.content_labels() == {"breathing_guidance"}
    legacy = EsSearch.parse_tags({"content_form_tags": [{"tag_id": "id1", "name": "呼吸引导"}]})
    assert legacy.content_labels() == {"breathing_guidance"}


def test_stage_filter_queries_existing_nested_code_field():
    assert _sleep_stage_filter(["入睡"]) == {
        "nested": {
            "path": "sleep_stage_tags",
            "query": {"terms": {"sleep_stage_tags.code": ["soothe"]}},
        }
    }


def test_unknown_chinese_tag_is_validation_error():
    with pytest.raises(ValueError):
        SearchAudioRequest(disliked_tags=["未审核的新标签"])


@pytest.mark.parametrize("values", ["rain", None, False, 1, [True], [{"name": "rain"}]])
def test_invalid_tag_container_is_rejected(values):
    with pytest.raises(ValueError):
        SearchAudioRequest(disliked_tags=values)


def test_future_code_wins_over_display_name():
    from app.core.audio_tags import reference_tag

    assert reference_tag({"code": "future_tag", "name": "呼吸引导"}) == "future_tag"


@pytest.mark.parametrize("code", ["none", "continuous", None])
def test_voice_engineering_filter_uses_dictionary_voice_code(code):
    from app.services.retrieval import ScoredCandidate, _apply_voice_code_filter

    source = {
        "content_form_tags": [{"code": "steady_light_rain", "name": "均匀小雨", "tag_id": "rain"}]
    }
    if code is not None:
        source["audio_engineering_tags"] = [{"code": "voice", "value": {"code": code}}]
    candidate = ScoredCandidate(source=source, tags=EsSearch.parse_tags(source))
    assert bool(_apply_voice_code_filter([candidate], disliked_tags=["voice"])) is (code == "none")


@pytest.mark.asyncio
async def test_unknown_chinese_rpc_tag_is_invalid_argument_not_ignored():
    from unittest.mock import MagicMock

    from app.core.config import Settings
    from app.server.somni.audio.catalog import AudioCatalogService, InvalidAudioQueryError

    service = AudioCatalogService(None, Settings(), retrieval=MagicMock())
    with pytest.raises(InvalidAudioQueryError):
        await service.search_prescription_audio({"disliked_tags": ["未审核的新标签"]})


def test_english_alias_does_not_match_inside_another_word():
    from app.services.retrieval import _match_labels_from_text

    rows = [{"code": "wind_ambience", "label": "wind_ambience", "name_en": "wind"}]
    assert _match_labels_from_text("unwind with music", rows) == []
    assert _match_labels_from_text("gentle wind", rows) == ["wind_ambience"]


@pytest.mark.asyncio
async def test_exact_exclusion_does_not_need_dictionary_vectors():
    from unittest.mock import MagicMock

    from app.core.config import Settings
    from app.services.retrieval import RetrievalService, ScoredCandidate

    source = {"content_form_tags": [{"name": "呼吸引导", "tag_id": "breathing"}]}
    candidate = ScoredCandidate(source=source, tags=EsSearch.parse_tags(source))
    service = RetrievalService(MagicMock(), MagicMock(), Settings())
    assert (
        await service._apply_dislike_filter(
            [candidate], ["breathing_guidance"], {}, dislike_vectors=[]
        )
        == []
    )


def test_get_audio_and_hot_tracking_share_english_tag_identity():
    from app.server.somni.audio.catalog import _has_content_form_code
    from app.server.somni.audio.hot import normalize_tag_code

    assert normalize_tag_code("人声引导") == "voice_guidance"
    assert normalize_tag_code("Voice Guidance") == "voice_guidance"
    assert _has_content_form_code(
        {"content_form_tags": [{"code": "voice_guidance", "name": "新展示名称"}]}, "voice_guidance"
    )
