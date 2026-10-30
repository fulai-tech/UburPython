"""RetrievalService 检索四步流水线单元测试（替身 EsSearch / Encoder）。"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.schemas.audio import SearchAudioRequest
from app.services.retrieval import (
    VOICE_MARKER,
    RetrievalService,
    ScoredCandidate,
    VectorSimilaritySnapshot,
    _apply_voice_code_filter,
    _cosine_similarity,
    _cosine_similarity_matrix,
    _match_labels_from_text,
    _normalize_to_dictionary_labels,
    _strip_voice_mention_tags,
    _tags_mention_voice,
    _usable_dislike_tags,
    _voice_value_code,
)

_VECTOR_DIM = 512


def test_vectorized_cosine_matches_scalar_reference() -> None:
    requests = [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.5, 2.0]]
    dictionary = [[3.0, 2.0, 1.0], [0.0, 0.0, 0.0], [1.0, -2.0, 0.5]]

    actual = _cosine_similarity_matrix(requests, dictionary)

    for row, request in enumerate(requests):
        for column, candidate in enumerate(dictionary):
            assert actual[row, column] == pytest.approx(
                _cosine_similarity(request, candidate),
                abs=1e-12,
            )


def test_vectorized_cosine_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        _cosine_similarity_matrix([[1.0, 0.0]], [[1.0, 0.0, 0.0]])


def test_similarity_snapshot_preserves_exclusive_tag_semantics() -> None:
    snapshot = VectorSimilaritySnapshot.build(
        [[1.0, 0.0], [1.0, 0.0]],
        {"steady_light_rain": [1.0, 0.0]},
    )

    assert (
        snapshot.count_matches(
            ["steady_light_rain"],
            ["white_noise", "steady_light_rain"],
            row_offset=0,
            threshold=0.8,
        )
        == 1
    )


def _tag_entries(prefix: str, labels: list[str]) -> list[dict[str, str]]:
    return [{"tag_id": f"{prefix}_{label}", "code": label, "name": label} for label in labels]


def _audio_doc(
    audio_name: str,
    *,
    doc_id: str | None = None,
    description_score: float | None = None,
    sleep_stage: list[str] | None = None,
    content_form: list[str] | None = None,
    mechanism: list[str] | None = None,
    audio_engineering: list[dict] | None = None,
) -> dict:
    doc = {
        "_id": doc_id or audio_name,
        "audio_url": f"https://cdn.example.com/{audio_name}.mp3",
        "audio_name": audio_name,
        "evidence_level_tags": [{"tag_id": "ev_B", "code": "B", "name": "B"}],
        "sleep_stage_tags": _tag_entries("ss", sleep_stage or []),
        "content_form_tags": _tag_entries("cf", content_form or []),
        "mechanism_tags": _tag_entries("mech", mechanism or []),
        "audio_engineering_tags": audio_engineering or [],
    }
    if description_score is not None:
        doc["_description_score"] = description_score
    return doc


def _voice_position(value_code: str, value_name: str) -> list[dict]:
    return [
        {
            "tag_id": "eng_voice_position",
            "code": "voice_position",
            "name": "voice_position",
            "value": {
                "tag_id": f"eng_{value_code}",
                "code": value_code,
                "name": value_name,
            },
        }
    ]


def _unit(index: int) -> list[float]:
    vec = [0.0] * _VECTOR_DIM
    vec[index] = 1.0
    return vec


def _build_service(
    es_search: MagicMock | None = None,
    encoder: MagicMock | None = None,
    settings: Settings | None = None,
) -> tuple[RetrievalService, MagicMock, MagicMock]:
    mock_es = es_search or MagicMock(spec=EsSearch)
    mock_es.list_content_tag_vectors = AsyncMock(return_value=[])
    mock_es.clear_content_tag_vectors_cache = MagicMock()
    mock_encoder = encoder or MagicMock(spec=Encoder)
    mock_encoder.encode = AsyncMock(
        side_effect=lambda texts: [[0.0] * _VECTOR_DIM for _ in texts],
    )
    mock_encoder.encode_one = AsyncMock(return_value=[0.0] * _VECTOR_DIM)
    svc = RetrievalService(
        mock_es,
        mock_encoder,
        settings or Settings(search_sleep_stage_filter_enabled=True),
    )
    return svc, mock_es, mock_encoder


@pytest.mark.asyncio
async def test_search_skips_sleep_stage_filter_when_disabled() -> None:
    """关闭睡眠阶段过滤时跳过步骤 1，直接拉全量候选进入内容形态准入。"""
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False),
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[_audio_doc("雨声A", sleep_stage=["wake"], content_form=["steady_light_rain"])]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain"],
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["雨声A"]
    es_search.list_all_audio_candidates.assert_awaited_once()
    es_search.filter_by_sleep_stage.assert_not_called()


@pytest.mark.asyncio
async def test_search_logs_four_step_timing_summary() -> None:
    """标签检索完成后输出四步耗时汇总，并标明最慢步骤。"""
    from loguru import logger

    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("雨声A", sleep_stage=["unwind"], content_form=["steady_light_rain"])
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_dictionary_vectors = AsyncMock(return_value={})
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"], content_tags=["steady_light_rain"], top_k=5
    )

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)))
    try:
        results = await service.search(request)
    finally:
        logger.remove(handler_id)

    assert [r["audio_name"] for r in results] == ["雨声A"]
    summary = [msg for msg in messages if "检索四步耗时" in msg]
    assert len(summary) == 1
    assert "最慢=" in summary[0]
    assert "步骤1睡眠阶段=" in summary[0]
    assert "步骤2内容准入=" in summary[0]
    assert "步骤3厌恶粗排=" in summary[0]
    assert "步骤4精排截断=" in summary[0]


@pytest.mark.asyncio
async def test_search_uses_sleep_stage_cache_when_available() -> None:
    """步骤1命中睡眠阶段缓存时不再打 ES。"""
    cached_doc = _audio_doc("缓存雨声", sleep_stage=["unwind"], content_form=["steady_light_rain"])
    sleep_cache = MagicMock()
    sleep_cache.get = AsyncMock(return_value=[cached_doc])
    service, es_search, _encoder = _build_service()
    service._sleep_stage_cache = sleep_cache
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_dictionary_vectors = AsyncMock(return_value={})
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"], content_tags=["steady_light_rain"], top_k=5
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["缓存雨声"]
    sleep_cache.get.assert_awaited_once_with(["unwind"])
    es_search.filter_by_sleep_stage.assert_not_called()


@pytest.mark.asyncio
async def test_search_cache_miss_uses_on_demand_stage_loader() -> None:
    cached_doc = _audio_doc("唤醒音乐", sleep_stage=["wake"], content_form=["music"])
    sleep_cache = MagicMock()
    sleep_cache.get = AsyncMock(return_value=None)
    sleep_cache.get_or_load = AsyncMock(return_value=[cached_doc])
    service, es_search, _encoder = _build_service()
    service._sleep_stage_cache = sleep_cache
    es_search.parse_tags = EsSearch.parse_tags

    results = await service.search(
        SearchAudioRequest(sleep_stage_tags=["wake"], content_tags=["music"], top_k=5)
    )

    assert [result["audio_name"] for result in results] == ["唤醒音乐"]
    sleep_cache.get_or_load.assert_awaited_once_with(
        ["wake"],
        service._load_sleep_stage_candidates,
    )
    es_search.filter_by_sleep_stage.assert_not_called()


@pytest.mark.asyncio
async def test_search_cache_loader_error_falls_back_to_es() -> None:
    es_doc = _audio_doc("ES 唤醒音乐", sleep_stage=["wake"], content_form=["music"])
    sleep_cache = MagicMock()
    sleep_cache.get = AsyncMock(return_value=None)
    sleep_cache.get_or_load = AsyncMock(side_effect=RuntimeError("Redis unavailable"))
    service, es_search, _encoder = _build_service()
    service._sleep_stage_cache = sleep_cache
    es_search.parse_tags = EsSearch.parse_tags
    es_search.filter_by_sleep_stage = AsyncMock(return_value=[es_doc])

    results = await service.search(
        SearchAudioRequest(sleep_stage_tags=["wake"], content_tags=["music"], top_k=5)
    )

    assert [result["audio_name"] for result in results] == ["ES 唤醒音乐"]
    es_search.filter_by_sleep_stage.assert_awaited_once_with(["wake"])


@pytest.mark.asyncio
async def test_search_merges_multi_stage_cache_via_service_path() -> None:
    """多睡眠阶段请求走缓存 get，由缓存侧按 audio_url 去重。"""
    sleep_cache = MagicMock()
    sleep_cache.get = AsyncMock(
        return_value=[
            _audio_doc("合并", sleep_stage=["unwind", "soothe"], content_form=["steady_light_rain"])
        ]
    )
    service, es_search, _encoder = _build_service()
    service._sleep_stage_cache = sleep_cache
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind", "soothe"],
        content_tags=["steady_light_rain"],
        top_k=5,
    )

    results = await service.search(request)

    assert len(results) == 1
    sleep_cache.get.assert_awaited_once_with(["unwind", "soothe"])
    es_search.filter_by_sleep_stage.assert_not_called()


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_sleep_stage_match() -> None:
    """睡眠阶段无命中时短路返回空，且不走向量编码。"""
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(return_value=[])
    request = SearchAudioRequest(sleep_stage_tags=["unwind"], content_tags=["steady_light_rain"])

    results = await service.search(request)

    assert results == []
    encoder.encode.assert_not_called()


@pytest.mark.asyncio
async def test_search_admits_on_exact_content_intersection() -> None:
    """内容标签精确交集命中时准入，match_count 为交集数量。"""
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(
                "雨声A", sleep_stage=["unwind"], content_form=["steady_light_rain", "forest_night"]
            ),
            _audio_doc("雨声B", sleep_stage=["unwind"], content_form=["steady_light_rain"]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain", "forest_night"],
        top_k=10,
    )

    es_search.get_dictionary_vectors = AsyncMock()
    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["雨声A", "雨声B"]
    encoder.encode.assert_not_awaited()
    es_search.get_dictionary_vectors.assert_not_called()


@pytest.mark.asyncio
async def test_search_falls_back_to_vector_when_no_exact_hit() -> None:
    """精确未命中时走向量模糊，余弦相似度 ≥ 阈值则准入。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("正念音频", sleep_stage=["unwind"], content_form=["mindfulness_test"])
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_dictionary_vectors = AsyncMock(return_value={"cf_mindfulness_test": unit_vec})
    encoder.encode = AsyncMock(return_value=[unit_vec])
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["mindfulness_sleep"],
        top_k=10,
    )

    results = await service.search(request)

    assert len(results) == 1
    assert results[0]["audio_name"] == "正念音频"
    encoder.encode.assert_awaited_once_with(["mindfulness_sleep"])


@pytest.mark.parametrize(
    ("requested_label", "candidate_label"),
    [
        ("white_noise", "pink_noise"),
        ("white_noise", "brown_noise"),
        ("pink_noise", "white_noise"),
        ("pink_noise", "brown_noise"),
        ("brown_noise", "white_noise"),
        ("brown_noise", "pink_noise"),
    ],
)
@pytest.mark.asyncio
async def test_search_does_not_fuzzy_match_between_color_noise_siblings(
    requested_label: str,
    candidate_label: str,
) -> None:
    """白、粉、棕噪音之间禁止经父标签或兄弟标签向量互相准入。"""
    unit_vec = _unit(0)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(
                candidate_label,
                sleep_stage=["unwind"],
                content_form=["color_noise", candidate_label],
            )
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={
            "cf_color_noise": unit_vec,
            f"cf_{candidate_label}": unit_vec,
        }
    )
    encoder.encode = AsyncMock(return_value=[unit_vec])
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=[requested_label],
        top_k=10,
    )

    results = await service.search(request)

    assert results == []


@pytest.mark.parametrize(
    ("requested_alias", "canonical", "sibling"),
    [
        ("pink_noise", "pink_noise", "brown_noise"),
        ("pink_noise", "pink_noise", "brown_noise"),
        ("brown_noise", "brown_noise", "pink_noise"),
        ("brown_noise", "brown_noise", "pink_noise"),
        ("white_noise", "white_noise", "pink_noise"),
    ],
)
@pytest.mark.asyncio
async def test_search_normalizes_color_noise_alias_and_excludes_siblings(
    requested_alias: str,
    canonical: str,
    sibling: str,
) -> None:
    """颜色噪音别名归一到规范标签后精确命中同色，且不召回兄弟色。"""
    unit_vec = _unit(0)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(canonical, sleep_stage=["unwind"], content_form=["color_noise", canonical]),
            _audio_doc(sibling, sleep_stage=["unwind"], content_form=["color_noise", sibling]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={
            "cf_color_noise": unit_vec,
            f"cf_{canonical}": unit_vec,
            f"cf_{sibling}": unit_vec,
        }
    )
    encoder.encode = AsyncMock(side_effect=lambda texts: [unit_vec for _ in texts])
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=[requested_alias],
        top_k=10,
    )

    results = await service.search(request)

    assert [result["audio_name"] for result in results] == [canonical]


@pytest.mark.parametrize(
    ("requested_alias", "canonical", "sibling"),
    [
        ("pink_noise", "pink_noise", "brown_noise"),
        ("brown_noise", "brown_noise", "pink_noise"),
        ("brown_noise", "brown_noise", "pink_noise"),
    ],
)
@pytest.mark.asyncio
async def test_text_query_normalizes_color_noise_alias_and_excludes_siblings(
    requested_alias: str,
    canonical: str,
    sibling: str,
) -> None:
    """自然语言输入颜色噪音别名时同样归一并隔离兄弟色。"""
    unit_vec = _unit(0)
    canonical_doc = _audio_doc(
        canonical, content_form=["color_noise", canonical], description_score=1.0
    )
    sibling_doc = _audio_doc(sibling, content_form=["color_noise", sibling], description_score=1.0)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": label, "dimension": "content_form", "vector": unit_vec}
            for label in ("color_noise", "white_noise", "pink_noise", "brown_noise")
        ]
    )
    es_search.search_by_description_vector = AsyncMock(return_value=[canonical_doc, sibling_doc])
    es_search.list_all_audio_candidates = AsyncMock(return_value=[canonical_doc, sibling_doc])
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={
            "cf_color_noise": unit_vec,
            f"cf_{canonical}": unit_vec,
            f"cf_{sibling}": unit_vec,
        }
    )
    encoder.encode_one = AsyncMock(return_value=unit_vec)
    encoder.encode = AsyncMock(side_effect=lambda texts: [unit_vec for _ in texts])

    results = await service.search(SearchAudioRequest(query_text=requested_alias, top_k=10))

    returned = {result["audio_name"] for result in results}
    assert sibling not in returned
    assert canonical in returned


@pytest.mark.asyncio
async def test_search_skips_content_admission_when_no_content_tags() -> None:
    """未传 content_tags 时保留睡眠阶段全集，且不走向量编码。"""
    service, es_search, encoder = _build_service()
    docs = [
        _audio_doc("音频A", sleep_stage=["unwind"], content_form=["steady_light_rain"]),
        _audio_doc("音频B", sleep_stage=["unwind"], content_form=["forest_night"]),
    ]
    es_search.filter_by_sleep_stage = AsyncMock(return_value=docs)
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(sleep_stage_tags=["unwind"], content_tags=[])

    results = await service.search(request)

    assert {r["audio_name"] for r in results} == {"音频A", "音频B"}
    encoder.encode.assert_not_called()


@pytest.mark.asyncio
async def test_search_keeps_different_codes_even_when_dislike_vectors_match() -> None:
    """不同标准类别不因向量相似而被硬剔除。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    orthogonal_vec = [0.0, 1.0] + [0.0] * (_VECTOR_DIM - 2)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("保留", sleep_stage=["unwind"], content_form=["steady_light_rain"]),
            _audio_doc("剔除", sleep_stage=["unwind"], content_form=["white_noise"]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    encoder.encode = AsyncMock(
        side_effect=lambda texts: [unit_vec for _ in texts],
    )
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={
            "cf_steady_light_rain": orthogonal_vec,
            "cf_white_noise": unit_vec,
        }
    )
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain", "white_noise"],
        disliked_tags=["heavy_rain"],
        top_k=10,
    )

    results = await service.search(request)

    assert {r["audio_name"] for r in results} == {"保留", "剔除"}
    encoder.encode.assert_awaited_once_with(["steady_light_rain", "white_noise", "heavy_rain"])
    es_search.get_dictionary_vectors.assert_awaited_once()
    assert set(es_search.get_dictionary_vectors.await_args.args[0]) == {
        "cf_steady_light_rain",
        "cf_white_noise",
    }


@pytest.mark.asyncio
async def test_search_keeps_candidate_when_disliked_vector_below_threshold() -> None:
    """厌恶标签向量与文档标签不相似时保留候选（不做精确字面剔除）。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    orthogonal_vec = [0.0, 1.0] + [0.0] * (_VECTOR_DIM - 2)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("白噪音音频", sleep_stage=["unwind"], content_form=["white_noise"])
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    encoder.encode = AsyncMock(
        side_effect=lambda texts: [
            unit_vec if t == "white_noise" else orthogonal_vec for t in texts
        ],
    )
    es_search.get_dictionary_vectors = AsyncMock(return_value={"cf_white_noise": unit_vec})
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["white_noise"],
        disliked_tags=["heavy_rain"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["白噪音音频"]


@pytest.mark.asyncio
async def test_search_vector_match_count_ranks_by_hit_count() -> None:
    """向量准入按每个 content_tag 独立计分，命中越多排序越靠前。"""
    unit_x = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    unit_y = [0.0, 1.0] + [0.0] * (_VECTOR_DIM - 2)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("单命中", sleep_stage=["unwind"], content_form=["rain_variant"]),
            _audio_doc(
                "双命中", sleep_stage=["unwind"], content_form=["rain_variant", "forest_variant"]
            ),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    encoder.encode = AsyncMock(return_value=[unit_x, unit_y])
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={"cf_rain_variant": unit_x, "cf_forest_variant": unit_y}
    )
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain", "forest_night"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["双命中", "单命中"]
    es_search.get_dictionary_vectors.assert_awaited_once()
    assert set(es_search.get_dictionary_vectors.await_args.args[0]) == {
        "cf_rain_variant",
        "cf_forest_variant",
    }


@pytest.mark.asyncio
async def test_search_coarse_rank_orders_by_match_count_desc() -> None:
    """粗排按命中标签数降序。"""
    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("少命中", sleep_stage=["unwind"], content_form=["steady_light_rain"]),
            _audio_doc(
                "多命中", sleep_stage=["unwind"], content_form=["steady_light_rain", "forest_night"]
            ),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain", "forest_night"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["多命中", "少命中"]


@pytest.mark.asyncio
async def test_content_admission_sets_match_count_without_tag_score() -> None:
    """步骤2准入只记录 match_count，粗排分 tag_score 留到厌恶剔除之后。"""
    service, es_search, _encoder = _build_service()
    es_search.parse_tags = EsSearch.parse_tags
    docs = [
        _audio_doc("少命中", sleep_stage=["unwind"], content_form=["steady_light_rain"]),
        _audio_doc(
            "多命中", sleep_stage=["unwind"], content_form=["steady_light_rain", "forest_night"]
        ),
    ]

    admitted = await service._apply_content_admission(
        docs, ["steady_light_rain", "forest_night"], {}
    )

    assert [(c.source["audio_name"], c.match_count, c.tag_score) for c in admitted] == [
        ("少命中", 1, 0.0),
        ("多命中", 2, 0.0),
    ]


@pytest.mark.asyncio
async def test_coarse_rank_runs_after_dislike_filter() -> None:
    """标签检索先厌恶剔除，再写 tag_score 粗排；被剔除候选不进入粗排。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    orthogonal_vec = [0.0, 1.0] + [0.0] * (_VECTOR_DIM - 2)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("保留少命中", sleep_stage=["unwind"], content_form=["steady_light_rain"]),
            _audio_doc(
                "剔除多命中",
                sleep_stage=["unwind"],
                content_form=["steady_light_rain", "forest_night"],
            ),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    encoder.encode = AsyncMock(side_effect=lambda texts: [unit_vec for _ in texts])
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={
            "cf_steady_light_rain": orthogonal_vec,
            "cf_forest_night": unit_vec,
        }
    )
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain", "forest_night"],
        disliked_tags=["forest_night"],
        top_k=10,
    )

    call_order: list[str] = []
    original_dislike = service._apply_dislike_filter
    original_coarse = service._apply_coarse_rank

    async def track_dislike(*args, **kwargs):
        call_order.append("dislike")
        return await original_dislike(*args, **kwargs)

    def track_coarse(candidates, *args, **kwargs):
        call_order.append("coarse")
        assert all(c.tag_score == 0.0 for c in candidates)
        return original_coarse(candidates, *args, **kwargs)

    service._apply_dislike_filter = track_dislike  # type: ignore[method-assign]
    service._apply_coarse_rank = track_coarse  # type: ignore[method-assign]

    results = await service.search(request)

    assert call_order == ["dislike", "coarse"]
    assert [r["audio_name"] for r in results] == ["保留少命中"]


@pytest.mark.asyncio
async def test_search_returns_all_when_top_k_omitted() -> None:
    """未传 top_k 时返回全部候选，不截断。"""
    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(f"音频{i}", sleep_stage=["unwind"], content_form=["steady_light_rain"])
            for i in range(5)
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain"],
    )

    results = await service.search(request)

    assert len(results) == 5


@pytest.mark.asyncio
async def test_search_caps_results_to_top_k() -> None:
    """精排截断到 top_k 条。"""
    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(f"音频{i}", sleep_stage=["unwind"], content_form=["steady_light_rain"])
            for i in range(5)
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain"],
        top_k=2,
    )

    results = await service.search(request)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_returns_projected_material() -> None:
    """标签检索只在最终返回前投影为 API 素材形状。"""
    service, es_search, _encoder = _build_service()
    doc = _audio_doc("雨声A", sleep_stage=["unwind"], content_form=["steady_light_rain"])
    doc["description"] = "描述"
    doc["content_form_tags"][0]["parent_tag_id"] = "p1"
    es_search.filter_by_sleep_stage = AsyncMock(return_value=[doc])
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(sleep_stage_tags=["unwind"], content_tags=["steady_light_rain"])

    results = await service.search(request)

    assert results[0] == {
        "_id": doc["_id"],
        "audio_name": "雨声A",
        "description": "描述",
        "audio_url": doc["audio_url"],
        "cover_url": "",
        "content_form_tags": [
            {
                "tag_id": "cf_steady_light_rain",
                "code": "steady_light_rain",
                "name": "steady_light_rain",
                "parent_tag_id": "p1",
            }
        ],
        "sleep_stage_tags": [{"tag_id": "ss_unwind", "code": "unwind", "name": "unwind"}],
        "audio_engineering_tags": [],
        "evidence_level_tags": [{"tag_id": "ev_B", "code": "B", "name": "B"}],
    }
    assert "status" not in results[0]


@pytest.mark.asyncio
async def test_text_query_can_return_description_only_recall() -> None:
    unit_vec = _unit(0)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(return_value=[])
    es_search.search_by_description_vector = AsyncMock(
        return_value=[
            _audio_doc(
                "描述命中",
                doc_id="desc-hit",
                description_score=0.9,
                content_form=["gentle_ocean_wave"],
            )
        ]
    )
    es_search.search_by_description_vector.return_value[0]["description"] = "海边声音"
    es_search.search_by_description_vector.return_value[0]["content_form_tags"][0][
        "parent_tag_id"
    ] = "p2"
    encoder.encode_one = AsyncMock(return_value=unit_vec)
    request = SearchAudioRequest(query_text="想要睡前舒缓的海边声音", top_k=10)

    results = await service.search(request)

    assert results[0] == {
        "_id": "desc-hit",
        "audio_name": "描述命中",
        "description": "海边声音",
        "audio_url": "https://cdn.example.com/描述命中.mp3",
        "cover_url": "",
        "content_form_tags": [
            {
                "tag_id": "cf_gentle_ocean_wave",
                "code": "gentle_ocean_wave",
                "name": "gentle_ocean_wave",
                "parent_tag_id": "p2",
            }
        ],
        "sleep_stage_tags": [],
        "audio_engineering_tags": [],
        "evidence_level_tags": [{"tag_id": "ev_B", "code": "B", "name": "B"}],
    }
    assert "status" not in results[0]
    es_search.search_by_description_vector.assert_awaited_once()
    es_search.list_all_audio_candidates.assert_not_called()


@pytest.mark.asyncio
async def test_text_query_recalls_description_and_tags_concurrently() -> None:
    unit_rain = _unit(0)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": "steady_light_rain", "dimension": "content_form", "vector": unit_rain}
        ]
    )
    encoder.encode_one = AsyncMock(return_value=unit_rain)
    encoder.encode = AsyncMock(return_value=[unit_rain])

    both_started = asyncio.Event()
    started: set[str] = set()

    async def recall(route: str) -> list[dict]:
        started.add(route)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.5)
        return []

    async def recall_description(*args, **kwargs) -> list[dict]:
        return await recall("description")

    async def recall_tags() -> list[dict]:
        return await recall("tags")

    es_search.search_by_description_vector = AsyncMock(side_effect=recall_description)
    es_search.list_all_audio_candidates = AsyncMock(side_effect=recall_tags)

    results = await service.search(SearchAudioRequest(query_text="steady_light_rain", top_k=10))

    assert results == []
    assert started == {"description", "tags"}


@pytest.mark.asyncio
async def test_text_query_recall_preserves_original_exception() -> None:
    unit_rain = _unit(0)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": "steady_light_rain", "dimension": "content_form", "vector": unit_rain}
        ]
    )
    encoder.encode_one = AsyncMock(return_value=unit_rain)
    expected = RuntimeError("description recall failed")
    es_search.search_by_description_vector = AsyncMock(side_effect=expected)
    es_search.list_all_audio_candidates = AsyncMock(return_value=[])

    with pytest.raises(RuntimeError) as caught:
        await service.search(SearchAudioRequest(query_text="steady_light_rain", top_k=10))

    assert caught.value is expected


@pytest.mark.asyncio
async def test_unresolved_query_exclusion_is_ignored_while_positive_recall_continues() -> None:
    unit_rain = _unit(0)
    unit_noise = _unit(1)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": "steady_light_rain", "dimension": "content_form", "vector": unit_rain},
            {
                "label": "noisy",
                "display_name": "嘈杂",
                "dimension": "content_form",
                "vector": unit_noise,
            },
        ]
    )
    es_search.search_by_description_vector = AsyncMock(return_value=[])
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("保留雨声", doc_id="keep", content_form=["steady_light_rain"]),
            _audio_doc("剔除嘈杂", doc_id="drop", content_form=["noisy"]),
        ]
    )
    encoder.encode_one = AsyncMock(return_value=unit_rain)
    encoder.encode = AsyncMock(
        side_effect=lambda texts: [
            unit_noise if t in {"noisy", "嘈杂"} else unit_rain for t in texts
        ]
    )
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={"cf_steady_light_rain": unit_rain, "cf_noisy": unit_noise}
    )
    request = SearchAudioRequest(query_text="睡前轻柔雨声，不要嘈杂", top_k=10)

    results = await service.search(request)

    assert [item["_id"] for item in results] == ["keep"]
    merged_calls = [call.args[0] for call in encoder.encode.await_args_list]
    assert ["睡前轻柔雨声，不要嘈杂"] in merged_calls
    assert ["嘈杂"] not in merged_calls
    assert ["睡前轻柔雨声，"] in merged_calls
    es_search.get_dictionary_vectors.assert_awaited()
    es_search.list_all_audio_candidates.assert_awaited_once()


@pytest.mark.parametrize(
    ("requested_label", "candidate_label"),
    [
        ("white_noise", "pink_noise"),
        ("white_noise", "brown_noise"),
        ("pink_noise", "white_noise"),
        ("pink_noise", "brown_noise"),
        ("brown_noise", "white_noise"),
        ("brown_noise", "pink_noise"),
    ],
)
@pytest.mark.asyncio
async def test_text_query_excludes_conflicting_color_noise_siblings(
    requested_label: str,
    candidate_label: str,
) -> None:
    """自然语言双路召回也不得在白、粉、棕噪音之间互相扩展。"""
    unit_vec = _unit(0)
    candidate = _audio_doc(
        candidate_label,
        sleep_stage=["unwind"],
        content_form=["color_noise", candidate_label],
    )
    candidate["_description_score"] = 1.0
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": label, "dimension": "content_form", "vector": unit_vec}
            for label in ("color_noise", "white_noise", "pink_noise", "brown_noise")
        ]
    )
    es_search.search_by_description_vector = AsyncMock(return_value=[candidate])
    es_search.list_all_audio_candidates = AsyncMock(return_value=[candidate])
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={
            "cf_color_noise": unit_vec,
            f"cf_{candidate_label}": unit_vec,
        }
    )
    encoder.encode_one = AsyncMock(return_value=unit_vec)
    encoder.encode = AsyncMock(side_effect=lambda texts: [unit_vec for _ in texts])

    results = await service.search(SearchAudioRequest(query_text=requested_label, top_k=10))

    assert results == []


@pytest.mark.asyncio
async def test_text_query_dislike_does_not_exclude_color_noise_sibling() -> None:
    """厌恶白噪音时，不得因向量相似而剔除粉噪音素材。"""
    rain_vec = _unit(0)
    noise_vec = _unit(1)
    candidate = _audio_doc(
        "粉噪音雨声",
        content_form=["steady_light_rain", "color_noise", "pink_noise"],
    )
    candidate["_description_score"] = 1.0
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": "steady_light_rain", "dimension": "content_form", "vector": rain_vec},
            {"label": "white_noise", "dimension": "content_form", "vector": noise_vec},
        ]
    )
    es_search.search_by_description_vector = AsyncMock(return_value=[candidate])
    es_search.list_all_audio_candidates = AsyncMock(return_value=[candidate])
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={
            "cf_steady_light_rain": rain_vec,
            "cf_color_noise": noise_vec,
            "cf_pink_noise": noise_vec,
        }
    )
    encoder.encode_one = AsyncMock(return_value=rain_vec)
    encoder.encode = AsyncMock(
        side_effect=lambda texts: [
            noise_vec if text == "white_noise" else rain_vec for text in texts
        ]
    )
    request = SearchAudioRequest(
        query_text="steady_light_rain",
        disliked_tags=["white_noise"],
        top_k=10,
    )

    results = await service.search(request)

    assert [result["audio_name"] for result in results] == ["粉噪音雨声"]


@pytest.mark.asyncio
async def test_text_query_fusion_ranks_two_route_hit_first() -> None:
    unit_rain = _unit(0)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": "steady_light_rain", "dimension": "content_form", "vector": unit_rain}
        ]
    )
    es_search.search_by_description_vector = AsyncMock(
        return_value=[
            _audio_doc(
                "双路命中", doc_id="both", description_score=0.8, content_form=["steady_light_rain"]
            ),
            _audio_doc(
                "仅描述命中",
                doc_id="desc",
                description_score=0.8,
                content_form=["gentle_ocean_wave"],
            ),
        ]
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("双路命中", doc_id="both", content_form=["steady_light_rain"]),
            _audio_doc("仅标签命中", doc_id="tag", content_form=["steady_light_rain"]),
        ]
    )
    encoder.encode_one = AsyncMock(return_value=unit_rain)
    encoder.encode = AsyncMock(return_value=[unit_rain])
    request = SearchAudioRequest(query_text="steady_light_rain", top_k=3)

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["双路命中", "仅描述命中", "仅标签命中"]


def test_match_labels_skips_short_and_prefers_longer() -> None:
    tags = [
        {"label": "l"},
        {"label": "n"},
        {"label": "low_dynamic", "display_name": "低动态"},
        {"label": "steady_light_rain"},
    ]
    matched = _match_labels_from_text("低动态无歌词均匀小雨", tags)
    assert "l" not in matched
    assert "n" not in matched
    assert "low_dynamic" in matched
    assert "steady_light_rain" in matched


def test_normalize_dislike_maps_natural_language_to_dictionary() -> None:
    tags = [
        {"label": "voice_guidance"},
        {"label": "white_noise"},
        {"label": "mechanical"},
    ]
    normalized = _normalize_to_dictionary_labels(
        [
            "voice",
            "mechanical",
            "unmapped_sentence",
            "white_noise",
        ],
        tags,
    )
    assert normalized == ["voice", "mechanical", "unmapped_sentence", "white_noise"]


@pytest.mark.asyncio
async def test_text_query_drops_conflicting_content_when_dislike_normalized() -> None:
    """显式 content 与归一后的厌恶标签冲突时，以厌恶为准剔除 content。"""
    unit_guide = _unit(0)
    unit_rain = _unit(1)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": "voice_guidance", "dimension": "content_form", "vector": unit_guide},
            {"label": "steady_light_rain", "dimension": "content_form", "vector": unit_rain},
        ]
    )
    es_search.search_by_description_vector = AsyncMock(
        return_value=[
            _audio_doc(
                "引导音", doc_id="guide", description_score=0.9, content_form=["voice_guidance"]
            ),
            _audio_doc(
                "雨声音频",
                doc_id="steady_light_rain",
                description_score=0.85,
                content_form=["steady_light_rain"],
            ),
        ]
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("引导音", doc_id="guide", content_form=["voice_guidance"]),
            _audio_doc("雨声音频", doc_id="steady_light_rain", content_form=["steady_light_rain"]),
        ]
    )
    es_search.get_dictionary_vectors = AsyncMock(
        return_value={"cf_voice_guidance": unit_guide, "cf_steady_light_rain": unit_rain}
    )
    encoder.encode = AsyncMock(
        side_effect=lambda texts: [
            unit_guide if ("语言" in t or "引导" in t) else unit_rain for t in texts
        ]
    )
    request = SearchAudioRequest(
        query_text="想要雨声",
        content_tags=["voice_guidance", "steady_light_rain"],
        disliked_tags=["voice_guidance"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["雨声音频"]


@pytest.mark.asyncio
async def test_encode_texts_reuses_lru_cache() -> None:
    service, _es_search, encoder = _build_service()
    encoder.encode = AsyncMock(return_value=[[1.0] + [0.0] * (_VECTOR_DIM - 1)])

    first = await service._encode_texts(["steady_light_rain"])
    second = await service._encode_texts(["steady_light_rain"])

    assert first == second
    encoder.encode.assert_awaited_once_with(["steady_light_rain"])


@pytest.mark.asyncio
async def test_warm_query_tag_vectors_encodes_dictionary_labels() -> None:
    """启动预热把内容词典 label 写入文本向量缓存，后续 encode 不再打模型。"""
    service, es_search, encoder = _build_service()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"id": "cf_steady_light_rain", "label": "steady_light_rain", "vector": [0.1]},
            {"id": "cf_rain_variant", "label": "rain_variant", "vector": [0.2]},
            {"id": "cf_dup", "label": "steady_light_rain", "vector": [0.3]},
        ]
    )
    encoder.encode = AsyncMock(
        side_effect=lambda texts: [
            [float(i)] + [0.0] * (_VECTOR_DIM - 1) for i, _ in enumerate(texts)
        ],
    )

    await service.warm_query_tag_vectors()
    encoder.encode.reset_mock()

    await service._encode_texts(["rain_variant", "steady_light_rain"])

    encoder.encode.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_query_tag_vectors_skips_when_dictionary_empty() -> None:
    service, es_search, encoder = _build_service()
    es_search.list_content_tag_vectors = AsyncMock(return_value=[])

    await service.warm_query_tag_vectors()

    encoder.encode.assert_not_awaited()


def test_vector_dislike_stays_soft_regardless_of_legacy_strong_threshold() -> None:
    """向量相似仅作软惩罚，不能绕过标准标签成为硬剔除。"""
    sim = 0.81
    dislike_vec = _unit(0)
    tag_vec = [0.0] * _VECTOR_DIM
    tag_vec[0] = sim
    tag_vec[1] = math.sqrt(1.0 - sim * sim)

    tags = EsSearch.parse_tags(_audio_doc("候选", content_form=["voice_position"]))
    dictionary = {"cf_voice_position": tag_vec}

    soft_service, _, _ = _build_service(settings=Settings(strong_dislike_sim_threshold=0.85))
    soft_penalty = soft_service._dislike_penalty(
        tags,
        disliked_tags=["voice"],
        dislike_vectors=[dislike_vec],
        dictionary_vectors=dictionary,
    )
    assert soft_penalty == 0.2

    hard_service, _, _ = _build_service(settings=Settings(strong_dislike_sim_threshold=0.78))
    hard_penalty = hard_service._dislike_penalty(
        tags,
        disliked_tags=["voice"],
        dislike_vectors=[dislike_vec],
        dictionary_vectors=dictionary,
    )
    assert hard_penalty == 0.2


def test_tags_mention_voice_detects_substring() -> None:
    assert _tags_mention_voice(["voice"]) is True
    assert _tags_mention_voice(["voice"]) is True
    assert _tags_mention_voice(["natural_sound", "music"]) is False


def test_strip_voice_mention_tags_removes_voice_related() -> None:
    assert _strip_voice_mention_tags(["voice", "sudden_events", "voice", "rhythm"]) == [
        "sudden_events",
        "rhythm",
    ]


def test_usable_dislike_tags_ignores_event_density() -> None:
    assert _usable_dislike_tags(["voice", "event_density", "sudden_events", "event_density"]) == [
        "voice",
        "sudden_events",
    ]


def test_voice_value_code_reads_nested_value() -> None:
    doc = _audio_doc("无人声轨", audio_engineering=_voice_position("none", "无人声"))
    assert _voice_value_code(doc) == "none"
    doc2 = _audio_doc(
        "有人声轨",
        audio_engineering=_voice_position("continuous", "人声贯穿大部分音频"),
    )
    assert _voice_value_code(doc2) == "continuous"
    assert _voice_value_code(_audio_doc("无字段")) is None


def test_voice_code_filter_dislike_keeps_none_only() -> None:
    none_doc = _audio_doc(
        "无人声",
        content_form=["steady_light_rain"],
        audio_engineering=_voice_position("none", "无人声"),
    )
    voice_doc = _audio_doc(
        "有人声",
        content_form=["steady_light_rain"],
        audio_engineering=_voice_position("intermittent", "人声间歇出现"),
    )
    missing_doc = _audio_doc("缺字段", content_form=["steady_light_rain"])
    candidates = [
        ScoredCandidate(source=none_doc, tags=EsSearch.parse_tags(none_doc)),
        ScoredCandidate(source=voice_doc, tags=EsSearch.parse_tags(voice_doc)),
        ScoredCandidate(source=missing_doc, tags=EsSearch.parse_tags(missing_doc)),
    ]
    kept = _apply_voice_code_filter(
        candidates,
        disliked_tags=["voice"],
    )
    assert [c.source["audio_name"] for c in kept] == ["无人声"]


def test_voice_code_filter_skips_when_dislike_has_no_voice() -> None:
    none_doc = _audio_doc(
        "无人声",
        audio_engineering=_voice_position("none", "无人声"),
    )
    voice_doc = _audio_doc(
        "有人声",
        audio_engineering=_voice_position("continuous", "人声贯穿大部分音频"),
    )
    candidates = [
        ScoredCandidate(source=none_doc, tags=EsSearch.parse_tags(none_doc)),
        ScoredCandidate(source=voice_doc, tags=EsSearch.parse_tags(voice_doc)),
    ]
    kept = _apply_voice_code_filter(
        candidates,
        disliked_tags=["heavy_rain"],
    )
    assert [c.source["audio_name"] for c in kept] == ["无人声", "有人声"]


@pytest.mark.asyncio
async def test_tag_search_dislike_voice_uses_code_not_vector() -> None:
    """dislike 含人声时按 value.code 留无人声，且不对人声做向量厌恶。"""
    unit_vec = _unit(0)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(
                "无人声雨",
                sleep_stage=["unwind"],
                content_form=["steady_light_rain"],
                audio_engineering=_voice_position("none", "无人声"),
            ),
            _audio_doc(
                "有人声雨",
                sleep_stage=["unwind"],
                content_form=["steady_light_rain"],
                audio_engineering=_voice_position("continuous", "人声贯穿大部分音频"),
            ),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_dictionary_vectors = AsyncMock(return_value={"cf_steady_light_rain": unit_vec})
    encoder.encode = AsyncMock(return_value=[unit_vec])
    request = SearchAudioRequest(
        sleep_stage_tags=["unwind"],
        content_tags=["steady_light_rain"],
        disliked_tags=["voice", "voice"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["无人声雨"]
    # 含「人声」的厌恶词已剥离，不会进入向量厌恶 encode
    for call in encoder.encode.await_args_list:
        assert all(VOICE_MARKER not in tag for tag in call.args[0])
