"""Bilingual constraints, unavailable vector services and rolling-cache safety."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.cache.sleep_stage_cache import SleepStageCandidateCache
from app.core.audio_tags import canonical_tag
from app.core.config import Settings
from app.es.search import EsSearch
from app.schemas.audio import SearchAudioRequest
from app.services.retrieval import RetrievalService, ScoredCandidate, _apply_voice_code_filter


def _service():
    es = MagicMock(spec=EsSearch)
    es.parse_tags = EsSearch.parse_tags
    es.list_content_tag_vectors = AsyncMock(return_value=[])
    es.get_dictionary_vectors = AsyncMock(return_value={})
    encoder = MagicMock()
    encoder.encode = AsyncMock(side_effect=lambda texts: [[0.0, 1.0] for _ in texts])
    return RetrievalService(es, encoder, Settings()), es, encoder


@pytest.mark.parametrize(
    "text",
    [
        "no voice",
        "without voice",
        "avoid voice",
        "I don't like voice",
        "不要人声",
        "不喜欢人声",
        "无人声",
    ],
)
async def test_negative_only_query_never_becomes_positive_voice(text):
    service, _, _ = _service()
    rows = [
        {
            "code": "voice",
            "label": "voice",
            "display_name": "人声",
            "name_en": "Vocals",
            "vector": [1.0, 0.0],
        }
    ]
    result = await service._extract_query_tags(text, [1.0, 0.0], rows)
    assert result.disliked_tags == ["voice"]
    assert result.content_tags == []


@pytest.mark.parametrize(
    "text",
    [
        "avoid voice, use steady light rain",
        "no voice but steady light rain",
        "不要人声，改用均匀小雨",
        "不喜欢人声改用均匀小雨",
    ],
)
async def test_negative_scope_keeps_positive_request(text):
    service, _, _ = _service()
    rows = [
        {"code": "voice", "label": "voice", "display_name": "人声", "vector": [1.0, 0.0]},
        {
            "code": "steady_light_rain",
            "label": "steady_light_rain",
            "display_name": "均匀小雨",
            "name_en": "Steady Light Rain",
            "vector": [0.0, 1.0],
        },
    ]
    result = await service._extract_query_tags(text, [1.0, 0.0], rows)
    assert result.disliked_tags == ["voice"]
    assert result.content_tags == ["steady_light_rain"]


async def test_exact_match_survives_vector_services_failure():
    service, es, encoder = _service()
    doc = {
        "_id": "rain",
        "audio_name": "Rain",
        "audio_url": "https://example.invalid/rain.mp3",
        "content_form_tags": [{"tag_id": "rain", "code": "steady_light_rain", "name": "均匀小雨"}],
    }
    es.filter_by_sleep_stage = AsyncMock(return_value=[doc])
    es.list_content_tag_vectors.side_effect = RuntimeError("dictionary offline")
    encoder.encode.side_effect = RuntimeError("encoder offline")
    result = await service.search(
        SearchAudioRequest(sleep_stage_tags=["soothe"], content_tags=["steady_light_rain"])
    )
    assert [item["_id"] for item in result] == ["rain"]


@pytest.mark.parametrize("values", [["none", "continuous"], ["continuous", "none"], ["none", None]])
def test_conflicting_voice_metadata_is_rejected(values):
    source = {
        "audio_engineering_tags": [{"code": "voice", "value": {"code": value}} for value in values]
    }
    candidate = ScoredCandidate(source=source, tags=EsSearch.parse_tags(source))
    assert _apply_voice_code_filter([candidate], disliked_tags=["voice"]) == []


@pytest.mark.parametrize("tag", ["🎤", "!!!", "", "  "])
def test_empty_or_symbol_free_text_is_ignored(tag):
    assert SearchAudioRequest(disliked_tags=[tag]).disliked_tags == []


@pytest.mark.parametrize("tag", ["风声", "温和闹铃音乐", "安静无声", "雨声"])
def test_unverified_or_ambiguous_alias_is_rejected(tag):
    with pytest.raises(ValueError):
        canonical_tag(tag)


def test_generic_piano_is_not_silently_narrowed_to_slow_piano():
    assert canonical_tag("piano") == "piano"


async def test_warm_preserves_previous_version_cache():
    from test_sleep_stage_cache import _FakeRedis

    redis = _FakeRedis()
    key = "sleep_stage_v2_index:入睡"
    await redis.set(key, json.dumps([]), ex=60)
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    await cache.warm(AsyncMock(return_value=[]))
    assert await redis.get(key) is not None


@pytest.mark.parametrize(
    "text", ["No Vocals", "I don’t want voice", "Please do not include voice", "free of voice"]
)
async def test_english_negative_variants(text):
    service, _, _ = _service()
    rows = [{"code": "voice", "name_en": "Vocals", "vector": [1.0, 0.0]}]
    result = await service._extract_query_tags(text, [1.0, 0.0], rows)
    assert result.disliked_tags == ["voice"]
    assert result.content_tags == []


def test_negation_has_english_word_boundaries():
    from app.services.retrieval import _extract_negative_fragments

    assert _extract_negative_fragments("noise with piano") == []
    assert _extract_negative_fragments("not only piano") == []


async def test_more_than_five_explicit_rejections_are_preserved():
    service, _, _ = _service()
    codes = [
        "voice",
        "body_scan",
        "breathing_guidance",
        "steady_light_rain",
        "dawn_chorus",
        "white_noise",
    ]
    result = await service._extract_query_tags(
        "avoid " + " and ".join(codes), [1.0, 0.0], [{"code": code} for code in codes]
    )
    assert set(result.disliked_tags) == set(codes)
    assert result.content_tags == []


async def test_bilingual_exact_rejection_works_when_vectors_are_offline():
    for label in ("呼吸引导", "breathing_guidance"):
        service, es, encoder = _service()
        es.filter_by_sleep_stage = AsyncMock(
            return_value=[
                {"_id": "forbidden", "content_form_tags": [{"code": "breathing_guidance"}]}
            ]
        )
        es.list_content_tag_vectors.side_effect = RuntimeError("offline")
        encoder.encode.side_effect = RuntimeError("offline")
        result = await service.search(
            SearchAudioRequest(sleep_stage_tags=["unwind"], disliked_tags=[label])
        )
        assert result == []


def test_alias_asset_is_derived_from_dictionary_fields():
    from app.core.audio_tags import build_dictionary_aliases, dictionary_records, tag_aliases

    rows = dictionary_records()
    assert tag_aliases() == build_dictionary_aliases(rows)
    assert len(rows) == 168  # Numeric test-stage codes 213/1321 are explicitly quarantined.
    for row in rows:
        assert canonical_tag(row["code"]) == row["code"]
    assert "风声" not in tag_aliases()
    assert "silence" not in tag_aliases()
    assert "gentle_alarm_music" not in tag_aliases()


async def test_result_cache_version_has_its_own_lru_namespace():
    from app.cache.audio_search_cache import AUDIO_SEARCH_KEY_PREFIX, AUDIO_SEARCH_LRU_INDEX_KEY

    assert AUDIO_SEARCH_KEY_PREFIX == "audio_search_v5+"
    assert AUDIO_SEARCH_LRU_INDEX_KEY == "audio_search_v5:_lru_index"


@pytest.mark.parametrize(
    "text",
    [
        "no voice, bird calls",
        "不要人声、鸟类叫声",
        "avoid " + "very " * 70 + "voice and bird calls",
    ],
)
async def test_negative_list_and_long_clause_do_not_leak_into_positive_tags(text):
    service, _, _ = _service()
    result = await service._extract_query_tags(
        text,
        [1.0, 0.0],
        [
            {"code": "voice", "name": "人声"},
            {"code": "bird_calls", "name": "鸟类叫声", "name_en": "Bird Calls"},
        ],
    )
    expected = {"bird_calls"} if "very " in text else {"voice", "bird_calls"}
    assert set(result.disliked_tags) == expected
    assert result.content_tags == []


async def test_explicit_parent_and_child_rejections_both_survive():
    service, _, _ = _service()
    rows = [{"code": "voice"}, {"code": "voice_guidance"}]
    result = await service._extract_query_tags("no voice and voice_guidance", [1.0, 0.0], rows)
    assert {"voice", "voice_guidance", "breathing_guidance", "wake_voice"} <= set(
        result.disliked_tags
    )


async def test_explicit_color_parent_rejection_is_not_removed_by_child():
    service, _, _ = _service()
    result = await service._extract_query_tags(
        "avoid color_noise and white_noise",
        [1.0, 0.0],
        [{"code": "color_noise"}, {"code": "white_noise"}],
    )
    assert set(result.disliked_tags) == {"color_noise", "white_noise"}


async def test_sentence_and_tag_vectors_cannot_pollute_each_other():
    service, es, _ = _service()
    es.list_content_tag_vectors.return_value = [{"code": "voice", "vector": [1.0, 0.0]}]
    assert await service._encode_one("voice") == [0.0, 1.0]
    assert await service._encode_texts(["voice"]) == [[1.0, 0.0]]
    assert await service._encode_one("voice") == [0.0, 1.0]


async def test_encoding_batch_larger_than_local_lru_does_not_lose_results():
    service, _, _ = _service()
    service._text_vector_cache_max = 1
    result = await service._encode_texts(["voice", "bird_calls"], natural=True)
    assert result == [[0.0, 1.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "query",
    [
        "不要库外类别",
        "avoid imaginary sounds",
        "不喜欢钢琴、库外类别",
        "avoid piano and imaginary sounds",
    ],
)
async def test_unknown_negative_query_does_not_short_circuit_recall(query):
    service, _, _ = _service()
    doc = {
        "_id": "stream",
        "audio_name": "Stream",
        "audio_url": "https://example.invalid/stream.mp3",
        "content_form_tags": [{"code": "stream"}],
    }
    service._recall_text_routes = AsyncMock(return_value=([doc], []))
    result = await service.search(SearchAudioRequest(query_text=query))
    service._recall_text_routes.assert_awaited_once()
    assert [x["_id"] for x in result] == ["stream"]
    extracted = await service._extract_query_tags(query, [0.0, 1.0], [])
    assert not extracted.unresolved_dislike
    assert extracted.disliked_tags == (
        ["slow_piano"] if "钢琴" in query or "piano" in query else []
    )
