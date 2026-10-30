"""HTTP searchAudio 空结果兜底：内容标签相似、描述向量、随机。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.schemas.audio import SearchAudioRequest
from app.services.retrieval import RetrievalService

_VECTOR_DIM = 3


def _tag_entries(prefix: str, labels: list[str]) -> list[dict[str, str]]:
    return [{"tag_id": f"{prefix}_{label}", "code": label, "name": label} for label in labels]


def _audio_doc(
    audio_name: str,
    *,
    sleep_stage: list[str] | None = None,
    content_form: list[str] | None = None,
    mechanism: list[str] | None = None,
) -> dict:
    return {
        "_id": audio_name,
        "audio_url": f"https://cdn.example.com/{audio_name}.mp3",
        "audio_name": audio_name,
        "sleep_stage_tags": _tag_entries("ss", sleep_stage or []),
        "content_form_tags": _tag_entries("cf", content_form or []),
        "mechanism_tags": _tag_entries("mech", mechanism or []),
        "audio_engineering_tags": [],
    }


def _dictionary(*rows: tuple[str, list[float]]) -> list[dict]:
    return [
        {
            "id": code,
            "code": code,
            "label": code,
            "display_name": code,
            "dimension": "content_form",
            "vector": vector,
        }
        for code, vector in rows
    ]


def _service(es_search: MagicMock | None = None) -> tuple[RetrievalService, MagicMock]:
    mock_es = es_search or MagicMock(spec=EsSearch)
    encoder = MagicMock(spec=Encoder)
    encoder.encode = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0, 0.0])
    service = RetrievalService(mock_es, encoder, Settings())
    return service, mock_es


@pytest.mark.asyncio
async def test_empty_content_tag_search_uses_nearest_tag_and_sleep_stage() -> None:
    service, es_search = _service()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=_dictionary(
            ("steady_light_rain", [1.0, 0.0, 0.0]),
            ("voice_guidance", [0.99, 0.01, 0.0]),
            ("stream", [0.8, 0.2, 0.0]),
            ("thunder", [0.0, 1.0, 0.0]),
        )
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("溪流加人声", sleep_stage=["unwind"], content_form=["stream", "voice_guidance"]),
            _audio_doc("溪流放松", sleep_stage=["unwind"], content_form=["stream"]),
            _audio_doc("溪流安抚", sleep_stage=["soothe"], content_form=["stream"]),
            _audio_doc("雷声放松", sleep_stage=["unwind"], content_form=["thunder"]),
        ]
    )

    results = await service.fallback_empty_search(
        SearchAudioRequest(
            sleep_stage_tags=["unwind"],
            content_tags=["steady_light_rain"],
            disliked_tags=["voice_guidance"],
        )
    )

    assert results.reason == "similar_tag"
    assert [item["audio_name"] for item in results.materials] == ["溪流放松"]
    es_search.search_by_description_vector.assert_not_called()


@pytest.mark.asyncio
async def test_similar_tag_with_only_disliked_audio_tries_the_next_tag() -> None:
    service, es_search = _service()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=_dictionary(
            ("steady_light_rain", [1.0, 0.0, 0.0]),
            ("stream", [0.8, 0.2, 0.0]),
            ("thunder", [0.0, 1.0, 0.0]),
        )
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("溪流加人声", sleep_stage=["unwind"], content_form=["stream", "voice_guidance"]),
            _audio_doc("雷声放松", sleep_stage=["unwind"], content_form=["thunder"]),
        ]
    )

    results = await service.fallback_empty_search(
        SearchAudioRequest(
            sleep_stage_tags=["unwind"],
            content_tags=["steady_light_rain"],
            disliked_tags=["voice_guidance"],
        )
    )

    assert results.reason == "similar_tag"
    assert [item["audio_name"] for item in results.materials] == ["雷声放松"]


@pytest.mark.asyncio
async def test_exhausted_similar_tags_returns_random_five_to_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    service, es_search = _service()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=_dictionary(("steady_light_rain", [1.0, 0.0, 0.0]))
    )
    pool = [_audio_doc(f"音频{index}") for index in range(12)]
    es_search.list_all_audio_candidates = AsyncMock(return_value=pool)
    monkeypatch.setattr("app.services.retrieval.random.randint", lambda _start, _end: 7)
    monkeypatch.setattr(
        "app.services.retrieval.random.sample",
        lambda items, count: items[:count],
    )

    results = await service.fallback_empty_search(
        SearchAudioRequest(content_tags=["steady_light_rain"])
    )

    assert results.reason == "random"
    assert [item["audio_name"] for item in results.materials] == [f"音频{index}" for index in range(7)]
    es_search.list_all_audio_candidates.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_text_fallback_uses_description_vector_limit() -> None:
    service, es_search = _service()
    hits = [
        _audio_doc("最相似", content_form=["stream"]),
        _audio_doc("次相似", content_form=["thunder"]),
        _audio_doc("厌恶", content_form=["voice_guidance"]),
    ]
    es_search.search_by_description_vector = AsyncMock(return_value=hits)

    default_results = await service.fallback_empty_search(
        SearchAudioRequest(query_text="海浪", disliked_tags=["voice_guidance"])
    )
    limited = await service.fallback_empty_search(
        SearchAudioRequest(query_text="海浪", disliked_tags=["voice_guidance"], top_k=1)
    )

    assert default_results.reason == "description"
    assert [item["audio_name"] for item in default_results.materials] == ["最相似", "次相似"]
    assert limited.reason == "description"
    assert [item["audio_name"] for item in limited.materials] == ["最相似"]
    first_call = es_search.search_by_description_vector.await_args_list[0]
    assert first_call.kwargs["size"] >= 400
    assert first_call.kwargs["sleep_stage_tags"] == []
    assert es_search.search_by_description_vector.await_args_list[1].kwargs["size"] >= 20


@pytest.mark.asyncio
async def test_similar_mechanism_tag_matches_audio_mechanism_field() -> None:
    service, es_search = _service()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "steady_light_rain",
                "code": "steady_light_rain",
                "label": "steady_light_rain",
                "display_name": "steady_light_rain",
                "dimension": "content_form",
                "vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "masking",
                "code": "masking",
                "label": "masking",
                "display_name": "masking",
                "dimension": "mechanism",
                "vector": [0.9, 0.1, 0.0],
            },
        ]
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[_audio_doc("遮蔽雨声", sleep_stage=["unwind"], mechanism=["masking"])]
    )

    results = await service.fallback_empty_search(
        SearchAudioRequest(sleep_stage_tags=["unwind"], content_tags=["steady_light_rain"])
    )

    assert results.reason == "similar_tag"
    assert [item["audio_name"] for item in results.materials] == ["遮蔽雨声"]


@pytest.mark.asyncio
async def test_dimension_mismatch_skips_tag_instead_of_failing() -> None:
    service, es_search = _service()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=_dictionary(
            ("steady_light_rain", [1.0, 0.0, 0.0]),
            ("stream", [0.99, 0.01]),
            ("thunder", [0.2, 0.8, 0.0]),
        )
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("溪流", content_form=["stream"]),
            _audio_doc("雷声", content_form=["thunder"]),
        ]
    )

    results = await service.fallback_empty_search(
        SearchAudioRequest(content_tags=["steady_light_rain"])
    )

    assert results.reason == "similar_tag"
    assert [item["audio_name"] for item in results.materials] == ["雷声"]


@pytest.mark.asyncio
async def test_description_fallback_prefers_requested_sleep_stage() -> None:
    service, es_search = _service()
    es_search.search_by_description_vector = AsyncMock(
        return_value=[
            _audio_doc("其他阶段", sleep_stage=["soothe"], content_form=["stream"]),
            _audio_doc("放松", sleep_stage=["unwind"], content_form=["thunder"]),
        ]
    )

    results = await service.fallback_empty_search(
        SearchAudioRequest(query_text="海浪", sleep_stage_tags=["unwind"])
    )

    assert results.reason == "description"
    assert [item["audio_name"] for item in results.materials] == ["放松"]


@pytest.mark.asyncio
async def test_content_tags_and_query_text_try_description_before_random() -> None:
    service, es_search = _service()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=_dictionary(("steady_light_rain", [1.0, 0.0, 0.0]))
    )
    es_search.list_all_audio_candidates = AsyncMock(return_value=[_audio_doc("无关")])
    es_search.search_by_description_vector = AsyncMock(
        return_value=[_audio_doc("描述命中", content_form=["stream"])]
    )

    results = await service.fallback_empty_search(
        SearchAudioRequest(content_tags=["steady_light_rain"], query_text="海浪")
    )

    assert results.reason == "description"
    assert [item["audio_name"] for item in results.materials] == ["描述命中"]


@pytest.mark.asyncio
async def test_query_text_without_hits_does_not_fall_through_to_random() -> None:
    service, es_search = _service()
    es_search.search_by_description_vector = AsyncMock(return_value=[])

    results = await service.fallback_empty_search(SearchAudioRequest(query_text="海浪"))

    assert results.materials == []
    assert results.reason is None
    es_search.list_all_audio_candidates.assert_not_called()
