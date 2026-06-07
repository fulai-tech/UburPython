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


def _tag_items(prefix: str, labels: list[str]) -> list[dict[str, str]]:
    return [{"vector_id": f"{prefix}_{label}", "label": label} for label in labels]


def _audio_doc(
    audio_name: str,
    *,
    sleep_stage: list[str] | None = None,
    content_form: list[str] | None = None,
    mechanism: list[str] | None = None,
) -> dict:
    return {
        "audio_url": f"https://cdn.example.com/{audio_name}.mp3",
        "audio_name": audio_name,
        "evidence_level": "B",
        "recommend_weight": 0.75,
        "tags": {
            "sleep_stage": _tag_items("ss", sleep_stage or []),
            "content_form": _tag_items("cf", content_form or []),
            "mechanism": _tag_items("mech", mechanism or []),
            "audio_feat": [],
            "rhythm": [],
            "risk_control": [],
        },
    }


def _build_service(
    es_search: MagicMock | None = None,
    encoder: MagicMock | None = None,
    settings: Settings | None = None,
) -> tuple[RetrievalService, MagicMock, MagicMock]:
    mock_es = es_search or MagicMock(spec=EsSearch)
    mock_encoder = encoder or MagicMock(spec=Encoder)
    svc = RetrievalService(mock_es, mock_encoder, settings or Settings())
    return svc, mock_es, mock_encoder


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

    es_search.get_tag_vectors = AsyncMock()
    results = await service.search(request)

    assert [r.audio_name for r in results] == ["雨声A", "雨声B"]
    encoder.encode.assert_awaited_once_with(["雨声", "森林"])
    es_search.get_tag_vectors.assert_not_called()


@pytest.mark.asyncio
async def test_search_falls_back_to_vector_when_no_exact_hit() -> None:
    """精确未命中时走向量模糊，余弦相似度 ≥ 阈值则准入。"""
    unit_vec = [1.0] + [0.0] * (_VECTOR_DIM - 1)
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[_audio_doc("正念音频", sleep_stage=["放松"], content_form=["正念"])]
    )
    es_search.parse_tags = EsSearch.parse_tags
    es_search.get_tag_vectors = AsyncMock(return_value={"cf_正念": unit_vec})
    encoder.encode = AsyncMock(return_value=[unit_vec])
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["冥想"],
        top_k=10,
    )

    results = await service.search(request)

    assert len(results) == 1
    assert results[0].audio_name == "正念音频"
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

    assert {r.audio_name for r in results} == {"音频A", "音频B"}
    encoder.encode.assert_not_called()


@pytest.mark.asyncio
async def test_search_removes_candidate_when_disliked_tag_intersects() -> None:
    """厌恶标签与内容标签有交集时剔除候选。"""
    service, es_search, encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[
            _audio_doc("保留", sleep_stage=["放松"], content_form=["雨声"]),
            _audio_doc("剔除", sleep_stage=["放松"], content_form=["白噪音"]),
        ]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(
        sleep_stage_tags=["放松"],
        content_tags=["雨声", "白噪音"],
        disliked_tags=["白噪音"],
        top_k=10,
    )

    results = await service.search(request)

    assert [r.audio_name for r in results] == ["保留"]


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

    assert [r.audio_name for r in results] == ["多命中", "少命中"]


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
async def test_search_result_tags_are_label_strings_only() -> None:
    """检索出参 tags 为 label 字符串列表，不含 vector_id。"""
    service, es_search, _encoder = _build_service()
    es_search.filter_by_sleep_stage = AsyncMock(
        return_value=[_audio_doc("雨声A", sleep_stage=["放松"], content_form=["雨声"])]
    )
    es_search.parse_tags = EsSearch.parse_tags
    request = SearchAudioRequest(sleep_stage_tags=["放松"], content_tags=["雨声"])

    results = await service.search(request)

    assert results[0].tags.sleep_stage == ["放松"]
    assert results[0].tags.content_form == ["雨声"]
