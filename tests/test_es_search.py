"""EsSearch 标签向量批量读取测试。"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.core.config import Settings
from app.es.search import EsSearch


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
