"""EsSearch 量产 GetAudio 双路检索查询测试。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.es.search import (
    GET_AUDIO_SOURCE_INCLUDES,
    EsSearch,
    build_get_audio_filters,
    locale_enabled_material_filter,
)


def test_get_audio_source_includes_material_fields() -> None:
    """原料索引回查只取状态/标签等；展示字段来自语言表 overlay。"""
    assert "status" in GET_AUDIO_SOURCE_INCLUDES
    assert "content_form_tags" in GET_AUDIO_SOURCE_INCLUDES
    assert "short_description" not in GET_AUDIO_SOURCE_INCLUDES
    assert "audio_name" not in GET_AUDIO_SOURCE_INCLUDES
    assert "audio_name_en" not in GET_AUDIO_SOURCE_INCLUDES


@pytest.mark.asyncio
async def test_locale_hydrate_keeps_engineering_and_evidence_tags() -> None:
    """语言表拼回原料时必须带上处方/播放依赖的工程与证据标签。"""
    client = MagicMock()
    client.search = AsyncMock(
        side_effect=[
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "loc1",
                            "_score": 1.2,
                            "_source": {
                                "material_id": "m1",
                                "language": "zh",
                                "audio_name": "雨",
                                "audio_url": "https://cdn/m1-zh.mp3",
                            },
                        }
                    ]
                }
            },
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "m1",
                            "_source": {
                                "status": True,
                                "content_form_tags": [{"code": "natural_sound"}],
                                "sleep_stage_tags": [{"code": "unwind"}],
                                "audio_engineering_tags": [
                                    {
                                        "code": "loopable",
                                        "value": {"code": "false"},
                                    }
                                ],
                                "evidence_level_tags": [{"code": "B"}],
                            },
                        }
                    ]
                }
            },
        ]
    )
    es = EsSearch(
        client,
        Settings(),
        index_profile="somni",
        material_locales_index="somni_audio_material_locales",
    )

    docs = await es.search_get_audio_bm25("雨", language="zh", size=5)

    assert docs[0]["_id"] == "m1"
    locale_body = client.search.await_args_list[0].kwargs["body"]
    assert locale_enabled_material_filter() in locale_body["query"]["bool"]["filter"]
    assert docs[0]["audio_engineering_tags"][0]["value"]["code"] == "false"
    assert docs[0]["evidence_level_tags"][0]["code"] == "B"
    material_body = client.search.await_args_list[1].kwargs["body"]
    includes = material_body["_source"]["includes"]
    assert "audio_engineering_tags" in includes
    assert "evidence_level_tags" in includes


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
async def test_ensure_indices_raises_on_field_type_conflict() -> None:
    """字段类型不能原地改时要失败，不能当成向量维度冲突跳过。"""
    from elasticsearch import ApiError

    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.create = AsyncMock()
    conflict = ApiError(
        message="mapper [description_text] cannot be changed from type [keyword] to [text]",
        meta=MagicMock(status=400),
        body={},
    )
    client.indices.put_mapping = AsyncMock(side_effect=conflict)
    es = EsSearch(client, Settings())

    with pytest.raises(ApiError):
        await es.ensure_indices()


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
    assert "description_vector" not in bodies[0]["mappings"]["properties"]
    assert "summary" not in bodies[0]["mappings"]["properties"]
    assert bodies[0]["mappings"]["properties"]["status"]["type"] == "boolean"
    assert bodies[1]["mappings"]["properties"]["name_en_vector"]["dims"] == 384
    assert bodies[1]["mappings"]["properties"]["language"]["type"] == "keyword"
    assert bodies[2]["mappings"]["properties"]["description_vector_en"]["dims"] == 384
    assert bodies[2]["mappings"]["properties"]["summary"]["type"] == "text"


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
