"""EsSearch 标签向量批量读取与检索候选查询测试。"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.core.config import Settings
from app.es.search import SEARCH_CANDIDATE_SOURCE_INCLUDES, EsSearch


@pytest.mark.asyncio
async def test_get_dictionary_vectors_deduplicates_and_batches_ids() -> None:
    client = MagicMock()
    client.mget = AsyncMock(
        side_effect=[
            {
                "docs": [
                    {"_id": "tag-a", "found": True, "_source": {"name_vector": [1.0]}},
                    {"_id": "tag-b", "found": False},
                ]
            },
            {"docs": [{"_id": "tag-c", "found": True, "_source": {"name_vector": [0.0]}}]},
        ]
    )
    search = EsSearch(
        client,
        Settings(es_dictionary_mget_batch_size=2),
    )

    result = await search.get_dictionary_vectors(["tag-a", "tag-b", "tag-a", "tag-c", ""])

    assert result == {"tag-a": [1.0], "tag-c": [0.0]}
    assert client.mget.await_args_list == [
        call(
            index=search.tag_dictionary_index,
            ids=["tag-a", "tag-b"],
            source_includes=["name_vector"],
        ),
        call(
            index=search.tag_dictionary_index,
            ids=["tag-c"],
            source_includes=["name_vector"],
        ),
    ]


@pytest.mark.asyncio
async def test_get_dictionary_vectors_skips_empty_ids() -> None:
    client = MagicMock()
    client.mget = AsyncMock()
    search = EsSearch(client, Settings())

    assert await search.get_dictionary_vectors([]) == {}
    assert await search.get_dictionary_vectors([""]) == {}
    client.mget.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_dictionary_vectors_reuses_process_cache() -> None:
    """同进程内已拉过的标签向量不再打 ES mget。"""
    client = MagicMock()
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "tag-a", "found": True, "_source": {"name_vector": [1.0]}},
                {"_id": "tag-b", "found": True, "_source": {"name_vector": [0.5]}},
            ]
        }
    )
    search = EsSearch(client, Settings())

    first = await search.get_dictionary_vectors(["tag-a", "tag-b"])
    second = await search.get_dictionary_vectors(["tag-b", "tag-a"])

    assert first == second == {"tag-a": [1.0], "tag-b": [0.5]}
    client.mget.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_dictionary_vectors_only_fetches_cache_misses() -> None:
    """缓存命中的 id 跳过，仅对未命中 id 发 mget。"""
    client = MagicMock()
    client.mget = AsyncMock(
        side_effect=[
            {
                "docs": [
                    {"_id": "tag-a", "found": True, "_source": {"name_vector": [1.0]}},
                ]
            },
            {
                "docs": [
                    {"_id": "tag-b", "found": True, "_source": {"name_vector": [0.2]}},
                ]
            },
        ]
    )
    search = EsSearch(client, Settings())

    await search.get_dictionary_vectors(["tag-a"])
    result = await search.get_dictionary_vectors(["tag-a", "tag-b"])

    assert result == {"tag-a": [1.0], "tag-b": [0.2]}
    assert client.mget.await_count == 2
    assert client.mget.await_args_list[1] == call(
        index=search.tag_dictionary_index,
        ids=["tag-b"],
        source_includes=["name_vector"],
    )


@pytest.mark.asyncio
async def test_clear_content_tag_vectors_cache_also_clears_dictionary_vectors() -> None:
    """词典同步失效时，内容列表缓存与按 id 向量缓存一并清空。"""
    client = MagicMock()
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "tag-a", "found": True, "_source": {"name_vector": [1.0]}},
            ]
        }
    )
    search = EsSearch(client, Settings())

    await search.get_dictionary_vectors(["tag-a"])
    search.clear_content_tag_vectors_cache()
    await search.get_dictionary_vectors(["tag-a"])

    assert client.mget.await_count == 2


@pytest.mark.asyncio
async def test_warm_dictionary_vectors_cache_seeds_from_content_tags() -> None:
    """启动预热把内容标签向量写入按 id 缓存，后续 get 不再 mget。"""
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "t1",
                        "_source": {
                            "name": "雨声",
                            "name_vector": [0.1, 0.2],
                            "type": "content_form",
                            "status": "启用",
                        },
                    }
                ]
            }
        }
    )
    client.mget = AsyncMock()
    search = EsSearch(client, Settings())

    await search.warm_dictionary_vectors_cache()
    result = await search.get_dictionary_vectors(["t1"])

    assert result == {"t1": [0.1, 0.2]}
    client.mget.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_content_tag_vectors_uses_process_cache() -> None:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "t1",
                        "_source": {
                            "name": "雨声",
                            "name_vector": [0.1],
                            "type": "content_form",
                            "status": "启用",
                        },
                    }
                ]
            }
        }
    )
    search = EsSearch(client, Settings())

    first = await search.list_content_tag_vectors()
    second = await search.list_content_tag_vectors()

    assert first == second
    assert first[0]["label"] == "雨声"
    client.search.assert_awaited_once()
    search.clear_content_tag_vectors_cache()
    await search.list_content_tag_vectors()
    assert client.search.await_count == 2


@pytest.mark.asyncio
async def test_filter_by_sleep_stage_uses_terms_and_includes_pipeline_fields() -> None:
    """步骤1用扁平 sleep_stage_names terms，并只取流水线必要字段。"""
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "a1",
                        "_source": {
                            "audio_name": "雨声",
                            "audio_url": "https://cdn.example.com/a.mp3",
                            "sleep_stage_tags": [
                                {"tag_id": "s1", "code": "unwind", "name": "放松"}
                            ],
                            "content_form_tags": [
                                {"tag_id": "c1", "code": "rain", "name": "雨声"}
                            ],
                        },
                    }
                ]
            }
        }
    )
    search = EsSearch(client, Settings())

    results = await search.filter_by_sleep_stage(["放松", "入睡"])

    assert results[0]["_id"] == "a1"
    assert "id" not in results[0]
    assert results[0]["audio_name"] == "雨声"
    assert "status" not in results[0]
    body = client.search.await_args.kwargs["body"]
    assert body["query"] == {"terms": {"sleep_stage_names": ["放松", "入睡"]}}
    assert body["_source"]["includes"] == list(SEARCH_CANDIDATE_SOURCE_INCLUDES)
    assert "status" not in body["_source"]["includes"]
    assert "description_vector" not in body["_source"]["includes"]
    assert "nested" not in str(body)


@pytest.mark.asyncio
async def test_filter_by_sleep_stage_empty_tags_skips_es() -> None:
    client = MagicMock()
    client.search = AsyncMock()
    search = EsSearch(client, Settings())

    assert await search.filter_by_sleep_stage([]) == []
    client.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_all_audio_candidates_includes_pipeline_fields() -> None:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "a1",
                        "_source": {
                            "audio_name": "雨声",
                            "recommend_weight": 80,
                        },
                    }
                ]
            }
        }
    )
    search = EsSearch(client, Settings())

    results = await search.list_all_audio_candidates()

    assert results[0]["_id"] == "a1"
    assert "id" not in results[0]
    body = client.search.await_args.kwargs["body"]
    assert body["query"] == {"match_all": {}}
    assert body["_source"]["includes"] == list(SEARCH_CANDIDATE_SOURCE_INCLUDES)
    assert "description_vector" not in body["_source"]["includes"]


@pytest.mark.asyncio
async def test_search_by_description_vector_includes_pipeline_fields() -> None:
    """description_vector knn 召回使用 includes 白名单，并保留 _description_score。"""
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "a1",
                        "_score": 0.87,
                        "_source": {
                            "audio_name": "雨声",
                            "description": "舒缓雨声",
                            "recommend_weight": 80,
                        },
                    }
                ]
            }
        }
    )
    search = EsSearch(client, Settings())
    query_vector = [0.1, 0.2, 0.3]

    results = await search.search_by_description_vector(
        query_vector,
        sleep_stage_tags=["放松"],
        size=10,
    )

    assert results[0]["_id"] == "a1"
    assert "id" not in results[0]
    assert results[0]["_description_score"] == 0.87
    body = client.search.await_args.kwargs["body"]
    assert body["_source"]["includes"] == list(SEARCH_CANDIDATE_SOURCE_INCLUDES)
    assert "description_vector" not in body["_source"]["includes"]
    assert body["knn"]["field"] == "description_vector"
    assert body["knn"]["query_vector"] == query_vector
