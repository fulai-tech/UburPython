"""量产音频目录查询。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AppError, EncoderNotReadyError
from app.server.somni.audio import catalog
from app.server.somni.audio.catalog import AudioCatalogService, InvalidAudioQueryError


class _Cursor:
    def __init__(self, docs: list) -> None:
        self._docs = list(docs)
        self._index = 0

    def skip(self, count: int) -> _Cursor:
        self._docs = self._docs[count:]
        return self

    def limit(self, count: int) -> _Cursor:
        self._docs = self._docs[:count]
        return self

    def __aiter__(self) -> _Cursor:
        return self

    async def __anext__(self):
        if self._index >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._index]
        self._index += 1
        return doc


_RAIN = {
    "_id": "a1",
    "audio_name": "雨夜",
    "language": "zh",
    "embedding": [0.1],
    "content_form_tags": [
        {
            "tag_id": "root-rain",
            "code": "natural_sound",
            "name": "自然声",
            "parent_tag_id": None,
        },
        {
            "tag_id": "child-rain",
            "code": "steady_rain",
            "name": "中雨/稳定雨声",
            "parent_tag_id": "root-rain",
        },
    ],
}
_MUSIC = {
    "_id": "a2",
    "audio_name": "钢琴",
    "language": "zh",
    "content_form_tags": [
        {
            "tag_id": "root-music",
            "code": "music",
            "name": "音乐",
            "parent_tag_id": None,
        }
    ],
}


def _service(
    collection: MagicMock,
    *,
    es_search: MagicMock | None = None,
    encoder: MagicMock | None = None,
    hot: MagicMock | None = None,
    fetch_all_hard_limit: int = 50,
    cache_ttl_sec: float = 60.0,
) -> AudioCatalogService:
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)
    settings = Settings(
        somni_mongo_db="Somni",
        somni_mongo_materials_collection="somni_audio_materials",
        default_page_size=1,
        max_page_size=200,
        fetch_all_hard_limit=fetch_all_hard_limit,
        get_audio_root_tag_sim_threshold=0.75,
        somni_audio_catalog_cache_ttl_sec=cache_ttl_sec,
    )
    return AudioCatalogService(
        client,
        settings,
        es_search=es_search,
        encoder=encoder,
        hot=hot,
    )


def _mongo_collection(docs: list) -> MagicMock:
    collection = MagicMock()

    def _match(query: dict) -> list:
        language = query.get("language")
        if language is None:
            return list(docs)
        return [doc for doc in docs if doc.get("language") == language]

    async def _count(query: dict) -> int:
        return len(_match(query))

    def _find(query: dict, *_args, **_kwargs) -> _Cursor:
        return _Cursor(_match(query))

    collection.count_documents = AsyncMock(side_effect=_count)
    collection.find = MagicMock(side_effect=_find)
    return collection


@pytest.mark.asyncio
async def test_get_audio_filters_by_language_mongo() -> None:
    zh_doc = {**_RAIN, "_id": "zh1", "language": "zh"}
    en_doc = {
        **_MUSIC,
        "_id": "en1",
        "audio_name": "Piano",
        "language": "en",
    }
    collection = _mongo_collection([zh_doc, en_doc])
    svc = _service(collection)

    zh_payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="zh",
    )
    en_payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="en",
    )

    assert [item["id"] for item in zh_payload["list"]] == ["zh1"]
    assert [item["id"] for item in en_payload["list"]] == ["en1"]
    assert collection.find.call_args_list[0].args[0] == {"language": "zh"}
    assert collection.find.call_args_list[1].args[0] == {"language": "en"}


@pytest.mark.asyncio
async def test_get_audio_keeps_language_caches_separate() -> None:
    zh_doc = {**_RAIN, "_id": "zh1", "language": "zh"}
    en_doc = {**_MUSIC, "_id": "en1", "language": "en"}
    collection = _mongo_collection([zh_doc, en_doc])
    svc = _service(collection)

    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code="", language="zh"
    )
    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code="", language="en"
    )

    assert collection.find.call_count == 2


@pytest.mark.asyncio
async def test_get_audio_filters_content_form_code_then_pages() -> None:
    svc = _service(_mongo_collection([_RAIN, _MUSIC]))
    payload = await svc.get_audio(
        page=1,
        page_size=1,
        fetch_all=False,
        query_text="",
        tag_code="steady_rain",
    )
    assert [item["id"] for item in payload["list"]] == ["a1"]
    assert payload["total"] == 1
    assert set(payload["list"][0]) == {
        "id",
        "audio_name",
        "audio_url",
        "cover_url",
        "description",
        "short_description",
        "vip",
        "tag",
    }
    assert payload["list"][0]["vip"] == 0
    assert payload["list"][0]["short_description"] == ""
    assert payload["list"][0]["tag"] == ["自然声", "中雨/稳定雨声"]


@pytest.mark.asyncio
async def test_get_audio_maps_short_description() -> None:
    doc = {**_RAIN, "short_description": "轻柔雨声助眠"}
    svc = _service(_mongo_collection([doc]))
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
    )
    assert payload["list"][0]["short_description"] == "轻柔雨声助眠"


@pytest.mark.asyncio
async def test_get_audio_tag_uses_name_en_when_language_en() -> None:
    doc = {
        **_RAIN,
        "language": "en",
        "content_form_tags": [
            {
                "tag_id": "root-rain",
                "code": "natural_sound",
                "name": "自然声",
                "name_en": "Natural Sound",
                "parent_tag_id": None,
            },
            {
                "tag_id": "child-rain",
                "code": "steady_rain",
                "name": "中雨/稳定雨声",
                "name_en": "Steady Rain",
                "parent_tag_id": "root-rain",
            },
        ],
    }
    svc = _service(_mongo_collection([doc]))
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="en",
    )
    assert payload["list"][0]["tag"] == ["Natural Sound", "Steady Rain"]


@pytest.mark.asyncio
async def test_get_audio_tag_empty_when_no_content_form_tags() -> None:
    doc = {
        "_id": "bare",
        "audio_name": "无标签",
        "language": "zh",
    }
    svc = _service(_mongo_collection([doc]))
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
    )
    assert payload["list"][0]["tag"] == []


@pytest.mark.asyncio
async def test_get_audio_tag_code_filter_is_case_insensitive() -> None:
    hot = MagicMock()
    hot.record_search = AsyncMock()
    svc = _service(_mongo_collection([_RAIN, _MUSIC]), hot=hot)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="Steady_Rain",
    )
    assert [item["id"] for item in payload["list"]] == ["a1"]
    await asyncio.sleep(0)
    hot.record_search.assert_awaited_once()
    assert hot.record_search.await_args.kwargs["tag_codes"] == {"steady_rain"}


@pytest.mark.asyncio
async def test_get_audio_uses_cache_on_second_call() -> None:
    collection = _mongo_collection([_RAIN, _MUSIC])
    svc = _service(collection)
    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="")
    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="music")
    assert collection.find.call_count == 1


@pytest.mark.asyncio
async def test_get_audio_keeps_mongo_and_es_caches_separate() -> None:
    collection = _mongo_collection([_RAIN, _MUSIC])
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN])
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "root-rain",
                "dimension": "content_form",
                "parent_tag_id": "",
                "vector": [1.0, 0.0],
            }
        ]
    )
    svc = _service(collection, es_search=es_search, encoder=encoder)

    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code=""
    )
    payload = await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="雨声", tag_code=""
    )

    es_search.list_audio_catalog_docs.assert_awaited_once_with(size=51, language="zh")
    assert [item["id"] for item in payload["list"]] == ["a1"]


@pytest.mark.asyncio
async def test_get_audio_cache_expires(monkeypatch) -> None:
    times = iter([100.0, 111.0])
    monkeypatch.setattr(catalog, "monotonic", lambda: next(times), raising=False)
    collection = _mongo_collection([_RAIN])
    svc = _service(collection, cache_ttl_sec=10.0)

    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code=""
    )
    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code=""
    )

    assert collection.find.call_count == 2


@pytest.mark.asyncio
async def test_get_audio_query_text_matches_root_tag_via_es() -> None:
    collection = _mongo_collection([_RAIN])
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "root-rain",
                "dimension": "content_form",
                "parent_tag_id": "",
                "vector": [1.0, 0.0],
            },
            {
                "id": "child-rain",
                "dimension": "content_form",
                "parent_tag_id": "root-rain",
                "vector": [1.0, 0.0],
            },
        ]
    )
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN, _MUSIC])
    svc = _service(collection, es_search=es_search, encoder=encoder)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
    )
    collection.find.assert_not_called()
    es_search.list_audio_catalog_docs.assert_awaited_once_with(size=51, language="zh")
    assert [item["id"] for item in payload["list"]] == ["a1"]


@pytest.mark.asyncio
async def test_get_audio_query_text_matches_child_tag() -> None:
    """搜索词更接近二级标签（如「白噪音」）时应按子标签命中，而非仅根标签。"""
    white_noise = {
        "_id": "a3",
        "audio_name": "白噪音",
        "content_form_tags": [
            {
                "tag_id": "root-color-noise",
                "code": "color_noise",
                "name": "颜色噪音",
                "parent_tag_id": None,
            },
            {
                "tag_id": "child-white-noise",
                "code": "white_noise",
                "name": "白噪音",
                "parent_tag_id": "root-color-noise",
            },
        ],
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "root-color-noise",
                "dimension": "content_form",
                "parent_tag_id": "",
                "vector": [0.0, 1.0],  # 与查询正交，根标签不命中
            },
            {
                "id": "child-white-noise",
                "dimension": "content_form",
                "parent_tag_id": "root-color-noise",
                "vector": [1.0, 0.0],  # 子标签命中
            },
        ]
    )
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN, white_noise])
    svc = _service(
        _mongo_collection([white_noise]),
        es_search=es_search,
        encoder=encoder,
    )
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="白噪音",
        tag_code="",
    )
    assert [item["id"] for item in payload["list"]] == ["a3"]


@pytest.mark.asyncio
async def test_get_audio_query_text_returns_empty_when_child_unused() -> None:
    """词典子标签命中但物料未挂该子标签时，直接返回空列表，不回退父级。"""
    pink = {
        "_id": "a4",
        "audio_name": "粉噪音",
        "content_form_tags": [
            {
                "tag_id": "root-color-noise",
                "code": "color_noise",
                "name": "颜色噪音",
                "parent_tag_id": None,
            },
            {
                "tag_id": "child-pink-noise",
                "code": "pink_noise",
                "name": "粉噪音",
                "parent_tag_id": "root-color-noise",
            },
        ],
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "root-color-noise",
                "dimension": "content_form",
                "parent_tag_id": "",
                "vector": [0.2, 0.8],
            },
            {
                "id": "child-white-noise",
                "dimension": "content_form",
                "parent_tag_id": "root-color-noise",
                "vector": [1.0, 0.0],
            },
        ]
    )
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN, pink])
    svc = _service(_mongo_collection([pink]), es_search=es_search, encoder=encoder)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="白噪音",
        tag_code="",
    )
    assert payload["list"] == []
    assert payload["total"] == 0


@pytest.mark.asyncio
async def test_get_audio_records_tag_hot_codes_when_query_hits() -> None:
    rain = {
        "_id": "a-rain",
        "audio_name": "Heavy Rain",
        "language": "en",
        "content_form_tags": [
            {
                "tag_id": "tag-heavy-rain",
                "code": "heavy_rain",
                "name": "大雨",
                "parent_tag_id": "root",
            }
        ],
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "tag-heavy-rain",
                "dimension": "content_form",
                "code": "heavy_rain",
                "label": "大雨",
                "name_en": "Heavy Rain",
                "parent_tag_id": "root",
                "vector": [0.0, 1.0],
                "vector_en": [0.0, 1.0],
            }
        ]
    )
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[rain])
    hot = MagicMock()
    hot.record_search = AsyncMock()
    svc = _service(
        _mongo_collection([rain]),
        es_search=es_search,
        encoder=encoder,
        hot=hot,
    )
    await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="rain",
        tag_code="",
        language="en",
    )
    await asyncio.sleep(0)
    hot.record_search.assert_awaited_once()
    kwargs = hot.record_search.await_args.kwargs
    assert kwargs["language"] == "en"
    assert kwargs["hit_count"] == 1
    assert kwargs["tag_codes"] == {"heavy_rain"}


@pytest.mark.asyncio
async def test_get_audio_records_hot_when_query() -> None:
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "root-rain",
                "dimension": "content_form",
                "code": "natural_sound",
                "parent_tag_id": "",
                "vector": [1.0, 0.0],
            }
        ]
    )
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN])
    hot = MagicMock()
    hot.record_search = AsyncMock()
    svc = _service(
        _mongo_collection([_RAIN]),
        es_search=es_search,
        encoder=encoder,
        hot=hot,
    )

    await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text=" 雨声 ",
        tag_code="",
    )
    await asyncio.sleep(0)

    hot.record_search.assert_awaited_once_with(
        " 雨声 ",
        language="zh",
        hit_count=1,
        tag_codes={"natural_sound"},
    )


@pytest.mark.asyncio
async def test_get_audio_succeeds_when_hot_recording_fails() -> None:
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "root-rain",
                "dimension": "content_form",
                "parent_tag_id": "",
                "vector": [1.0, 0.0],
            }
        ]
    )
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN])
    hot = MagicMock()
    hot.record_search = AsyncMock(side_effect=RuntimeError("hot unavailable"))
    svc = _service(
        _mongo_collection([_RAIN]),
        es_search=es_search,
        encoder=encoder,
        hot=hot,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
    )
    await asyncio.sleep(0)

    assert [item["id"] for item in payload["list"]] == ["a1"]
    hot.record_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_audio_query_text_without_encoder_fails() -> None:
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN])
    svc = _service(_mongo_collection([_RAIN]), es_search=es_search)
    with pytest.raises(EncoderNotReadyError):
        await svc.get_audio(
            page=1,
            page_size=10,
            fetch_all=False,
            query_text="雨声",
            tag_code="",
        )


@pytest.mark.asyncio
async def test_get_audio_rejects_invalid_page() -> None:
    svc = _service(_mongo_collection([]))
    with pytest.raises(InvalidAudioQueryError):
        await svc.get_audio(
            page=0,
            page_size=20,
            fetch_all=False,
            query_text="",
            tag_code="",
        )


@pytest.mark.asyncio
async def test_get_audio_mongo_load_rejects_over_limit() -> None:
    collection = MagicMock()
    collection.count_documents = AsyncMock(return_value=6)
    svc = _service(collection, fetch_all_hard_limit=5)
    with pytest.raises(InvalidAudioQueryError):
        await svc.get_audio(
            page=None,
            page_size=None,
            fetch_all=False,
            query_text="",
            tag_code="",
        )


@pytest.mark.asyncio
async def test_get_audio_tag_requires_mongo() -> None:
    svc = AudioCatalogService(None, Settings())
    with pytest.raises(AppError) as exc:
        await svc.get_audio_tag()
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_get_audio_tag_maps_root_fields() -> None:
    collection = MagicMock()
    collection.count_documents = AsyncMock(return_value=1)
    collection.find = MagicMock(
        return_value=_Cursor(
            [
                {
                    "_id": "root-natural",
                    "type": "content_form",
                    "code": "natural_sound",
                    "name": "自然声",
                    "name_en": "Natural Sound",
                    "parent_tag_id": None,
                    "status": "启用",
                }
            ]
        )
    )
    svc = _service(collection)
    payload = await svc.get_audio_tag()
    assert payload["tags"][0] == {
        "type": "content_form",
        "code": "natural_sound",
        "name": "自然声",
        "name_en": "Natural Sound",
        "id": "root-natural",
        "parent_tag_id": None,
        "status": "启用",
    }
    query = catalog._root_tag_query()
    assert query["type"] == "content_form"
    assert query["status"] == "启用"
    collection.find.assert_called_once_with(
        query,
        {
            "_id": 1,
            "id": 1,
            "type": 1,
            "code": 1,
            "name": 1,
            "name_en": 1,
            "parent_tag_id": 1,
            "status": 1,
        },
    )


def test_root_tag_query_only_content_form() -> None:
    query = catalog._root_tag_query()
    assert query["type"] == "content_form"


def test_lexical_match_content_tag_matches_code_token_and_name_en() -> None:
    tag = {
        "code": "heavy_rain",
        "label": "大雨",
        "name_en": "Heavy Rain",
    }
    assert catalog._lexical_match_content_tag(tag, "rain") is True
    assert catalog._lexical_match_content_tag(tag, "Heavy Rain") is True
    assert catalog._lexical_match_content_tag(tag, "heavy_rain") is True
    assert catalog._lexical_match_content_tag(tag, "piano") is False


def test_tag_vector_for_language_prefers_en_then_fallback() -> None:
    tag = {"vector": [1.0], "vector_en": [2.0]}
    assert catalog._tag_vector_for_language(tag, "en") == [2.0]
    assert catalog._tag_vector_for_language(tag, "zh") == [1.0]
    assert catalog._tag_vector_for_language({"vector": [1.0]}, "en") == [1.0]


@pytest.mark.asyncio
async def test_get_audio_query_text_rain_matches_via_code_token() -> None:
    rain = {
        "_id": "a-rain",
        "audio_name": "Heavy Rain Loop",
        "language": "en",
        "content_form_tags": [
            {
                "tag_id": "tag-heavy-rain",
                "code": "heavy_rain",
                "name": "大雨",
                "parent_tag_id": "root",
            }
        ],
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "tag-heavy-rain",
                "dimension": "content_form",
                "code": "heavy_rain",
                "label": "大雨",
                "name_en": "Heavy Rain",
                "parent_tag_id": "root",
                "vector": [0.0, 1.0],
                "vector_en": [0.0, 1.0],
            }
        ]
    )
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[rain])
    svc = _service(
        _mongo_collection([rain]),
        es_search=es_search,
        encoder=encoder,
    )
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="rain",
        tag_code="",
        language="en",
    )
    assert [item["id"] for item in payload["list"]] == ["a-rain"]


def test_to_vip_normalizes_bool_int_and_string() -> None:
    assert catalog._to_vip(None) == 0
    assert catalog._to_vip(False) == 0
    assert catalog._to_vip(0) == 0
    assert catalog._to_vip("false") == 0
    assert catalog._to_vip("0") == 0
    assert catalog._to_vip(True) == 1
    assert catalog._to_vip(1) == 1
    assert catalog._to_vip("true") == 1
    assert catalog._to_vip("1") == 1


@pytest.mark.asyncio
async def test_get_audio_maps_vip_true_to_one() -> None:
    doc = {**_RAIN, "vip": True}
    svc = _service(_mongo_collection([doc]))
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
    )
    assert payload["list"][0]["vip"] == 1


@pytest.mark.asyncio
async def test_drain_hot_tasks_waits_for_pending_recording() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_record(*_args, **_kwargs):
        started.set()
        await release.wait()

    hot = MagicMock()
    hot.record_search = AsyncMock(side_effect=_slow_record)
    svc = _service(_mongo_collection([_RAIN]), hot=hot)
    svc._schedule_hot("雨声", "zh", 1)

    await started.wait()
    release.set()
    await svc.drain_hot_tasks(timeout_sec=1.0)

    assert not svc._hot_tasks
    hot.record_search.assert_awaited_once_with(
        "雨声", language="zh", hit_count=1, tag_codes=set()
    )
