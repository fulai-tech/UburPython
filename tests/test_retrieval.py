"""RetrievalService 检索四步流水线单元测试（替身 EsSearch / Encoder）。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.schemas.audio import SearchAudioRequest
from app.services.retrieval import RetrievalService

_VECTOR_DIM = 512


def _tag_entries(prefix: str, labels: list[str]) -> list[dict[str, str]]:
    return [
        {"tag_id": f"{prefix}_{label}", "code": label, "name": label}
        for label in labels
    ]


def _audio_doc(
    audio_name: str,
    *,
    doc_id: str | None = None,
    description_score: float | None = None,
    sleep_stage: list[str] | None = None,
    content_form: list[str] | None = None,
    mechanism: list[str] | None = None,
) -> dict:
    doc = {
        "_id": doc_id or audio_name,
        "audio_url": f"https://cdn.example.com/{audio_name}.mp3",
        "audio_name": audio_name,
        "evidence_level_tags": [{"tag_id": "ev_B", "code": "B", "name": "中等证据"}],
        "sleep_stage_tags": _tag_entries("ss", sleep_stage or []),
        "content_form_tags": _tag_entries("cf", content_form or []),
        "mechanism_tags": _tag_entries("mech", mechanism or []),
        "audio_engineering_tags": [],
        "medical_risk_tags": [],
    }
    if description_score is not None:
        doc["_description_score"] = description_score
    return doc


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
    mock_encoder = encoder or MagicMock(spec=Encoder)
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
        return_value=[_audio_doc("雨声A", sleep_stage=["清醒"], content_form=["雨声"])]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声"],
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["雨声A"]
    es_search.list_all_audio_candidates.assert_awaited_once()
    es_search.filter_by_sleep_stage.assert_not_called()


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_sleep_stage_match() -> None:
    """睡眠阶段无命中时短路返回空，且不走向量编码。"""
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(return_value=[])
    request = SearchAudioRequest(sleep_stage_tags=["放松"], content_tags=["雨声"])

    results = await service.search(request)

    assert results == []
    encoder.encode.assert_not_called()


@pytest.mark.asyncio
async def test_search_admits_on_exact_content_intersection() -> None:
    """内容标签精确交集命中时准入，match_count 为交集数量。"""
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("雨声A", sleep_stage=["放松"], content_form=["雨声", "森林"]),
            _audio_doc("雨声B", sleep_stage=["放松"], content_form=["雨声"]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声", "森林"],
        top_k=10,
    )

    es_search.get_dictionary_vectors = AsyncMock()
    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["雨声A", "雨声B"]
    encoder.encode.assert_awaited_once_with(["雨声", "森林"])
    es_search.get_dictionary_vectors.assert_not_called()


@pytest.mark.asyncio
async def test_search_falls_back_to_vector_when_no_exact_hit() -> None:
    """精确未命中时走向量模糊，余弦相似度 ≥ 阈值则准入。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[_audio_doc("正念音频", sleep_stage=["放松"], content_form=["正念"])]
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_dictionary_vectors = AsyncMock(return_value={"cf_正念": unit_vec})
    encoder.encode = AsyncMock(return_value=[unit_vec])
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["冥想"],
        top_k=10,
    )

    results = await service.search(request)

    assert len(results) == 1
    assert results[0]["audio_name"] == "正念音频"
    encoder.encode.assert_awaited_once_with(["冥想"])


@pytest.mark.asyncio
async def test_search_skips_content_admission_when_no_content_tags() -> None:
    """未传 content_tags 时保留睡眠阶段全集，且不走向量编码。"""
    service, es_search, encoder = _build_service()
    docs = [
        _audio_doc("音频A", sleep_stage=["放松"], content_form=["雨声"]),
        _audio_doc("音频B", sleep_stage=["放松"], content_form=["森林"]),
    ]
    es_search.filter_by_sleep_stage = AsyncMock(return_value=docs)
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(sleep_stage_tags=["放松"], content_tags=[])

    results = await service.search(request)

    assert {r["audio_name"] for r in results} == {"音频A", "音频B"}
    encoder.encode.assert_not_called()


@pytest.mark.asyncio
async def test_search_removes_candidate_when_disliked_vector_matches() -> None:
    """厌恶标签向量与文档内容标签向量余弦 ≥ 阈值时剔除候选。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    orthogonal_vec = [0.0, 1.0] + [0.0] * (_VECTOR_DIM - 2)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("保留", sleep_stage=["放松"], content_form=["雨声"]),
            _audio_doc("剔除", sleep_stage=["放松"], content_form=["白噪音"]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    encoder.encode = AsyncMock(
        side_effect=[
            [unit_vec, unit_vec],
            [unit_vec],
        ]
    )
    es_search.get_dictionary_vectors = AsyncMock(
        side_effect=[
            {"cf_雨声": orthogonal_vec},
            {"cf_白噪音": unit_vec},
        ]
    )
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声", "白噪音"],
        disliked_tags=["嘈杂"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["保留"]
    assert encoder.encode.await_args_list[1].args[0] == ["嘈杂"]


@pytest.mark.asyncio
async def test_search_keeps_candidate_when_disliked_vector_below_threshold() -> None:
    """厌恶标签向量与文档标签不相似时保留候选（不做精确字面剔除）。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    orthogonal_vec = [0.0, 1.0] + [0.0] * (_VECTOR_DIM - 2)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[_audio_doc("白噪音音频", sleep_stage=["放松"], content_form=["白噪音"])]
    )
    es_search.parse_tags = EsSearch.parse_tags
    encoder.encode = AsyncMock(
        side_effect=[
            [unit_vec],
            [orthogonal_vec],
        ]
    )
    es_search.get_dictionary_vectors = AsyncMock(return_value={"cf_白噪音": unit_vec})
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["白噪音"],
        disliked_tags=["白噪音"],
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
            _audio_doc("单命中", sleep_stage=["放松"], content_form=["下雨声"]),
            _audio_doc("双命中", sleep_stage=["放松"], content_form=["下雨声", "大森林"]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    encoder.encode = AsyncMock(return_value=[unit_x, unit_y])
    es_search.get_dictionary_vectors = AsyncMock(
        side_effect=[
            {"cf_下雨声": unit_x},
            {"cf_下雨声": unit_x, "cf_大森林": unit_y},
        ]
    )
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声", "森林"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["双命中", "单命中"]


@pytest.mark.asyncio
async def test_search_coarse_rank_orders_by_match_count_desc() -> None:
    """粗排按命中标签数降序。"""
    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("少命中", sleep_stage=["放松"], content_form=["雨声"]),
            _audio_doc("多命中", sleep_stage=["放松"], content_form=["雨声", "森林"]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声", "森林"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["多命中", "少命中"]


@pytest.mark.asyncio
async def test_search_returns_all_when_top_k_omitted() -> None:
    """未传 top_k 时返回全部候选，不截断。"""
    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(f"音频{i}", sleep_stage=["放松"], content_form=["雨声"])
            for i in range(5)
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声"],
    )

    results = await service.search(request)

    assert len(results) == 5


@pytest.mark.asyncio
async def test_search_caps_results_to_top_k() -> None:
    """精排截断到 top_k 条。"""
    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc(f"音频{i}", sleep_stage=["放松"], content_form=["雨声"])
            for i in range(5)
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声"],
        top_k=2,
    )

    results = await service.search(request)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_returns_full_material_document() -> None:
    """检索出参为 somni_audio_materials 索引文档，不做字段裁剪。"""
    service, es_search, _encoder = _build_service()
    doc = _audio_doc("雨声A", sleep_stage=["放松"], content_form=["雨声"])
    doc["id"] = "6a33a7928030d4cf420efeb6"
    es_search.filter_by_sleep_stage = AsyncMock(return_value=[doc])
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(sleep_stage_tags=["放松"], content_tags=["雨声"])

    results = await service.search(request)

    assert results[0]["id"] == "6a33a7928030d4cf420efeb6"
    assert results[0]["audio_name"] == "雨声A"
    assert results[0]["sleep_stage_tags"][0]["name"] == "放松"
    assert results[0]["content_form_tags"][0]["name"] == "雨声"
    assert results[0]["evidence_level_tags"][0]["code"] == "B"


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
                content_form=["海浪"],
            )
        ]
    )
    encoder.encode_one = AsyncMock(return_value=unit_vec)
    request = SearchAudioRequest(query_text="想要睡前舒缓的海边声音", top_k=10)

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["描述命中"]
    es_search.search_by_description_vector.assert_awaited_once()
    es_search.list_all_audio_candidates.assert_not_called()


@pytest.mark.asyncio
async def test_text_query_extracts_positive_and_negative_tags() -> None:
    unit_rain = _unit(0)
    unit_noise = _unit(1)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"label": "雨声", "dimension": "content_form", "vector": unit_rain},
            {"label": "嘈杂", "dimension": "content_form", "vector": unit_noise},
        ]
    )
    es_search.search_by_description_vector = AsyncMock(return_value=[])
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("保留雨声", doc_id="keep", content_form=["雨声"]),
            _audio_doc("剔除嘈杂", doc_id="drop", content_form=["嘈杂"]),
        ]
    )
    encoder.encode_one = AsyncMock(return_value=unit_rain)
    encoder.encode = AsyncMock(
        side_effect=lambda texts: [unit_rain if t == "雨声" else unit_noise for t in texts]
    )
    es_search.get_dictionary_vectors = AsyncMock(return_value={"cf_嘈杂": unit_noise})
    request = SearchAudioRequest(query_text="睡前轻柔雨声，不要嘈杂", top_k=10)

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["保留雨声"]
    assert ["雨声"] in [call.args[0] for call in encoder.encode.await_args_list]
    assert ["嘈杂"] in [call.args[0] for call in encoder.encode.await_args_list]


@pytest.mark.asyncio
async def test_text_query_fusion_ranks_two_route_hit_first() -> None:
    unit_rain = _unit(0)
    service, es_search, encoder = _build_service(
        settings=Settings(search_sleep_stage_filter_enabled=False)
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[{"label": "雨声", "dimension": "content_form", "vector": unit_rain}]
    )
    es_search.search_by_description_vector = AsyncMock(
        return_value=[
            _audio_doc("双路命中", doc_id="both", description_score=0.8, content_form=["雨声"]),
            _audio_doc("仅描述命中", doc_id="desc", description_score=0.8, content_form=["海浪"]),
        ]
    )
    es_search.list_all_audio_candidates = AsyncMock(
        return_value=[
            _audio_doc("双路命中", doc_id="both", content_form=["雨声"]),
            _audio_doc("仅标签命中", doc_id="tag", content_form=["雨声"]),
        ]
    )
    encoder.encode_one = AsyncMock(return_value=unit_rain)
    encoder.encode = AsyncMock(return_value=[unit_rain])
    request = SearchAudioRequest(query_text="雨声", top_k=3)

    results = await service.search(request)

    assert [r["audio_name"] for r in results] == ["双路命中", "仅描述命中", "仅标签命中"]
