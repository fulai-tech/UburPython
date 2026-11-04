"""EsSearch 量产 GetAudio 双路检索查询测试。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.es.search import (
    GET_AUDIO_SOURCE_INCLUDES,
    EsSearch,
    build_get_audio_filters,
)


def test_get_audio_source_includes_short_description() -> None:
    assert "short_description" in GET_AUDIO_SOURCE_INCLUDES
    assert "short_description_en" in GET_AUDIO_SOURCE_INCLUDES
    assert "description" not in GET_AUDIO_SOURCE_INCLUDES
    assert "audio_name_en" in GET_AUDIO_SOURCE_INCLUDES


def test_build_get_audio_filters_with_tag() -> None:
    filters = build_get_audio_filters(tag_code=" steady_rain ")

    assert filters == [
        {"term": {"status": True}},
        {
            "nested": {
                "path": "content_form_tags",
                "query": {"term": {"content_form_tags.code": "steady_rain"}},
            }
        },
    ]
    assert not any("language" in str(item) for item in filters)


def test_build_get_audio_filters_without_tag() -> None:
    filters = build_get_audio_filters()

    assert filters == [
        {"term": {"status": True}},
    ]


@pytest.mark.asyncio
async def test_search_get_audio_bm25_uses_zh_description_text() -> None:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {"_id": "a1", "_score": 2.0, "_source": {"audio_name": "雨"}}
                ]
            }
        }
    )
    es = EsSearch(client, Settings())

    docs = await es.search_get_audio_bm25(
        "雨声",
        language="zh",
        tag_code="",
        size=30,
    )

    assert docs == [{"_id": "a1", "audio_name": "雨", "_bm25_score": 2.0}]
    body = client.search.await_args.kwargs["body"]
    assert body["query"]["bool"]["must"][0]["match_phrase"]["description_text"] == "雨声"
    assert {"term": {"status": True}} in body["query"]["bool"]["filter"]
    assert not any(
        item.get("term", {}).get("language") for item in body["query"]["bool"]["filter"]
    )


@pytest.mark.asyncio
async def test_search_get_audio_bm25_uses_en_description_text() -> None:
    client = MagicMock()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    es = EsSearch(client, Settings())

    await es.search_get_audio_bm25(
        "rain",
        language="en",
        size=10,
    )

    body = client.search.await_args.kwargs["body"]
    assert body["query"]["bool"]["must"][0]["match_phrase"]["description_text_en"] == "rain"


@pytest.mark.asyncio
async def test_search_get_audio_knn_uses_description_vector_en() -> None:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [{"_id": "a1", "_score": 0.9, "_source": {"language": "en"}}]
            }
        }
    )
    es = EsSearch(client, Settings())
    query_vector = [0.1] * 8

    docs = await es.search_get_audio_knn(
        query_vector,
        language="en",
        tag_code="music",
        k=50,
        num_candidates=150,
    )

    assert docs == [{"_id": "a1", "language": "en", "_knn_score": 0.9}]
    body = client.search.await_args.kwargs["body"]
    assert body["knn"]["field"] == "description_vector_en"
    assert body["knn"]["query_vector"] == query_vector
    assert {"exists": {"field": "description_vector_en"}} in body["knn"]["filter"]
    assert not any(
        item.get("term", {}).get("language") for item in body["knn"]["filter"]
    )


def test_build_get_audio_filters_with_content_form_codes() -> None:
    filters = build_get_audio_filters(
        content_form_codes=["white_noise", "pink_noise"],
    )
    nested = filters[-1]["nested"]["query"]["bool"]["should"]
    assert {"term": {"content_form_tags.code": "white_noise"}} in nested
    assert {"term": {"content_form_tags.code": "pink_noise"}} in nested


def test_build_get_audio_filters_tag_code_overrides_intent_codes() -> None:
    filters = build_get_audio_filters(
        tag_code="music",
        content_form_codes=["white_noise"],
    )
    assert filters[-1] == {
        "nested": {
            "path": "content_form_tags",
            "query": {"term": {"content_form_tags.code": "music"}},
        }
    }


@pytest.mark.asyncio
async def test_search_get_audio_vector_all_uses_script_score_and_vector_filter() -> None:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [{"_id": "a1", "_score": 0.9, "_source": {"language": "zh"}}]
            }
        }
    )
    es = EsSearch(client, Settings())
    query_vector = [0.1] * 8

    docs = await es.search_get_audio_vector_all(
        query_vector,
        language="zh",
        size=100,
    )

    assert docs == [{"_id": "a1", "language": "zh", "_knn_score": 0.9}]
    body = client.search.await_args.kwargs["body"]
    script_score = body["query"]["script_score"]
    assert script_score["script"] == {
        "source": "(cosineSimilarity(params.q, 'description_vector') + 1.0) / 2",
        "params": {"q": query_vector},
    }
    assert {"exists": {"field": "description_vector"}} in script_score["query"]["bool"][
        "filter"
    ]


@pytest.mark.asyncio
async def test_ensure_indices_skips_dense_vector_dims_conflict() -> None:
    """手板旧索引 name_en_vector dims 冲突时跳过，不拖垮启动。"""
    from elasticsearch import ApiError

    client = MagicMock()
    client.indices.exists = AsyncMock(side_effect=[True, True])
    client.indices.create = AsyncMock()
    conflict = ApiError(
        message="Mapper for [name_en_vector] conflicts with existing mapper:\n\tCannot update parameter [dims] from [512] to [384]",
        meta=MagicMock(status=400),
        body={},
    )
    client.indices.put_mapping = AsyncMock(side_effect=[None, conflict])
    es = EsSearch(client, Settings())

    await es.ensure_indices()

    assert client.indices.create.await_count == 0
    assert client.indices.put_mapping.await_count == 2


@pytest.mark.asyncio
async def test_ensure_indices_handboard_profile_uses_handboard_mapping() -> None:
    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    es = EsSearch(client, Settings(embedding_dim=512, embedding_dim_en=384))

    await es.ensure_indices()

    bodies = [c.kwargs["body"] for c in client.indices.create.await_args_list]
    tag_props = bodies[1]["mappings"]["properties"]
    assert tag_props["name_en_vector"]["dims"] == 512
    assert "description_vector_en" not in bodies[0]["mappings"]["properties"]


@pytest.mark.asyncio
async def test_ensure_indices_somni_profile_uses_bilingual_mapping() -> None:
    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    es = EsSearch(
        client,
        Settings(embedding_dim=512, embedding_dim_en=384),
        index_profile="somni",
    )

    await es.ensure_indices()

    bodies = [c.kwargs["body"] for c in client.indices.create.await_args_list]
    assert bodies[0]["mappings"]["properties"]["description_vector_en"]["dims"] == 384
    assert bodies[1]["mappings"]["properties"]["name_en_vector"]["dims"] == 384


@pytest.mark.asyncio
async def test_search_get_audio_bm25_long_en_uses_should_phrases() -> None:
    client = MagicMock()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    es = EsSearch(client, Settings())

    await es.search_get_audio_bm25(
        "soft piano for bedtime",
        language="en",
        size=10,
    )

    must = client.search.await_args.kwargs["body"]["query"]["bool"]["must"][0]
    assert "bool" in must
    phrases = [
        list(item["match_phrase"].values())[0] for item in must["bool"]["should"]
    ]
    assert "piano" in phrases
    assert "bedtime" in phrases
