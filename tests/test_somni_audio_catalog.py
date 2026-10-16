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
    encoder_en: MagicMock | None = None,
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
        encoder_en=encoder_en,
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
        "vip",
        "tag",
    }
    assert payload["list"][0]["vip"] == 0
    assert payload["list"][0]["description"] == ""
    assert payload["list"][0]["tag"] == ["自然声", "中雨/稳定雨声"]


@pytest.mark.asyncio
async def test_get_audio_maps_short_description_into_description() -> None:
    doc = {
        **_RAIN,
        "description": "长描述不应返回",
        "short_description": "轻柔雨声助眠",
    }
    svc = _service(_mongo_collection([doc]))
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
    )
    assert payload["list"][0]["description"] == "轻柔雨声助眠"
    assert "short_description" not in payload["list"][0]


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
async def test_get_audio_cache_stays_resident(monkeypatch) -> None:
    times = iter([100.0, 100_000.0])
    monkeypatch.setattr(catalog, "monotonic", lambda: next(times), raising=False)
    collection = _mongo_collection([_RAIN])
    svc = _service(collection, cache_ttl_sec=0)

    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code=""
    )
    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code=""
    )

    assert collection.find.call_count == 1


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
    docs = [{"_id": str(i), "language": "zh"} for i in range(6)]
    svc = _service(_mongo_collection(docs), fetch_all_hard_limit=5)
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


@pytest.mark.asyncio
async def test_prescription_search_resolves_codes_and_uses_ranked_search_without_hot(monkeypatch) -> None:
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[{"id": "a1", "audio_url": "https://cdn/a.mp3", "_score": 0.9, "sleep_stage_tags": [{"code": "unwind"}]}])
    service = _service(MagicMock(), hot=MagicMock())
    service._retrieval = retrieval

    class Cursor:
        def __init__(self, projection):
            document = {
                "code": "breathing_guidance", "type": "content_form", "name": "呼吸引导"
            }
            self.document = {
                key: value for key, value in document.items() if projection.get(key)
            }

        def __aiter__(self):
            self.items = iter([self.document])
            return self
        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    tags = MagicMock()
    tags.find.side_effect = lambda query, projection: Cursor(projection)
    monkeypatch.setattr(service, "_tags", lambda: tags)
    result = await service.search_prescription_audio({
        "query_text": "gentle breathing", "sleep_stage_tags": ["unwind"],
        "content_tag_codes": ["breathing_guidance"], "content_tags": ["rain"],
        "disliked_tags": ["thunder"], "top_k": 5,
    })
    assert result[0]["id"] == "a1"
    request = retrieval.search.await_args.args[0]
    assert request.sleep_stage_tags == ["unwind"]
    assert request.content_tags == ["呼吸引导", "rain"]
    assert request.disliked_tags == ["thunder"]
    service._hot.record_search.assert_not_called()


@pytest.mark.asyncio
async def test_prescription_search_fails_closed_on_unknown_tag_code(monkeypatch) -> None:
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[{"id": "must-not-return"}])
    service = _service(MagicMock())
    service._retrieval = retrieval

    class EmptyCursor:
        def __aiter__(self): return self
        async def __anext__(self): raise StopAsyncIteration

    tags = MagicMock()
    tags.find.return_value = EmptyCursor()
    monkeypatch.setattr(service, "_tags", lambda: tags)
    assert await service.search_prescription_audio({"query_text": "rain", "content_tag_codes": ["missing_code"]}) == []
    retrieval.search.assert_not_called()


@pytest.mark.asyncio
async def test_prescription_search_fails_closed_on_ambiguous_tag_code(monkeypatch) -> None:
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[{"id": "must-not-return"}])
    service = _service(MagicMock())
    service._retrieval = retrieval

    class Cursor:
        def __aiter__(self):
            self.items = iter([
                {"code": "shared", "type": "content_form", "name": "相同名称"},
                {"code": "shared", "type": "mechanism", "name": "相同名称"},
            ])
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    tags = MagicMock()
    tags.find.return_value = Cursor()
    monkeypatch.setattr(service, "_tags", lambda: tags)

    assert await service.search_prescription_audio({"query_text": "rain", "content_tag_codes": ["shared"]}) == []
    retrieval.search.assert_not_called()


@pytest.mark.asyncio
async def test_get_audio_orders_quiz_preference_by_created_at_then_rest() -> None:
    older_natural = {**_RAIN, "_id": "old-rain", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-12-01T00:00:00Z"}
    newer_natural = {
        **_RAIN,
        "_id": "new-rain",
        "audio_name": "新雨",
        "created_at": "2026-03-01T00:00:00Z",
    }
    vector_hit = {
        **_MUSIC,
        "_id": "wave",
        "audio_name": "海浪",
        "created_at": "2026-02-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "sea", "code": "wave", "name": "海浪"}],
    }
    other = {
        "_id": "piano",
        "audio_name": "钢琴新",
        "language": "zh",
        "created_at": "2026-06-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "v", "code": "voice_guidance", "name": "人声引导"}],
    }
    encoder = MagicMock()
    encoder.is_loaded = True

    async def _encode_zh(text: str) -> list[float]:
        return [1.0, 0.0] if text == "海浪" else [0.0, 1.0]

    encoder.encode_one = AsyncMock(side_effect=_encode_zh)
    es_search = MagicMock()
    es_search.search_by_description_vector = AsyncMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "sea",
                "label": "海浪",
                "name_en": "Wave",
                "dimension": "content_form",
                "vector": [1.0, 0.0],
            }
        ]
    )
    svc = _service(
        _mongo_collection([older_natural, other, vector_hit, newer_natural]),
        es_search=es_search,
        encoder=encoder,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=["自然声"],
        preference_text="海浪拍岸",
    )

    assert [item["id"] for item in payload["list"]] == ["new-rain", "wave", "old-rain", "piano"]
    assert payload["total"] == 4
    es_search.search_by_description_vector.assert_not_called()
    encoded = [call.args[0] for call in encoder.encode_one.await_args_list]
    assert "海浪" in encoded
    assert "海浪拍岸" not in encoded

    page2 = await svc.get_audio(
        page=2,
        page_size=2,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=["自然声"],
        preference_text="海浪拍岸",
    )
    assert [item["id"] for item in page2["list"]] == ["old-rain", "piano"]


@pytest.mark.asyncio
async def test_get_audio_same_created_at_keeps_catalog_order() -> None:
    first = {**_RAIN, "_id": "first", "created_at": "2026-01-01T00:00:00Z"}
    second = {**_RAIN, "_id": "second", "audio_name": "同样新", "created_at": "2026-01-01T00:00:00Z"}
    svc = _service(_mongo_collection([first, second]))

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=["自然声"],
    )

    assert [item["id"] for item in payload["list"]] == ["first", "second"]


@pytest.mark.asyncio
async def test_get_audio_content_form_name_matches_level1_label() -> None:
    voice = {
        "_id": "voice",
        "audio_name": "人声引导",
        "language": "zh",
        "created_at": "2026-01-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "v", "code": "voice_guidance", "name": "人声引导"}],
    }
    same_code_other_name = {
        "_id": "alias",
        "audio_name": "语言引导",
        "language": "zh",
        "created_at": "2026-08-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "v2", "code": "voice_guidance", "name": "语言引导"}],
    }
    other = {**_RAIN, "_id": "rain", "created_at": "2026-06-01T00:00:00Z"}
    svc = _service(_mongo_collection([other, same_code_other_name, voice]))

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=["人声引导"],
    )

    assert [item["id"] for item in payload["list"]] == ["voice", "alias", "rain"]


@pytest.mark.asyncio
async def test_get_audio_dislike_text_moves_matching_tags_to_end() -> None:
    steady = {
        "_id": "steady",
        "audio_name": "风扇",
        "language": "zh",
        "created_at": "2026-07-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "s", "code": "steady", "name": "稳态背景音"}],
    }
    rain = {
        "_id": "rain",
        "audio_name": "雨",
        "language": "zh",
        "created_at": "2026-08-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "r", "code": "rain", "name": "中雨/稳定雨声"}],
    }
    piano = {
        "_id": "piano",
        "audio_name": "钢琴曲",
        "language": "zh",
        "created_at": "2026-06-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "p", "code": "piano", "name": "慢钢琴"}],
    }
    other = {
        "_id": "other",
        "audio_name": "其他",
        "language": "zh",
        "created_at": "2026-03-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "o", "code": "other", "name": "鸟鸣"}],
    }
    encoder = MagicMock()
    encoder.is_loaded = True

    async def _encode_zh(text: str) -> list[float]:
        return {
            "雨声": [1.0, 0.0, 0.0],
            "钢琴": [0.0, 1.0, 0.0],
        }.get(text, [0.0, 0.0, 1.0])

    encoder.encode_one = AsyncMock(side_effect=_encode_zh)
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {"id": "r", "label": "中雨/稳定雨声", "dimension": "content_form", "vector": [1.0, 0.0, 0.0]},
            {"id": "p", "label": "慢钢琴", "dimension": "content_form", "vector": [0.0, 1.0, 0.0]},
        ]
    )
    svc = _service(_mongo_collection([other, rain, piano, steady]), es_search=es_search, encoder=encoder)

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=["稳态背景音"],
        dislike_text="小提琴雨声钢琴声",
    )

    assert [item["id"] for item in payload["list"]] == ["steady", "other", "rain", "piano"]


@pytest.mark.asyncio
async def test_get_audio_english_like_matches_en_name_and_dislike_matches_english_tag() -> None:
    wave = {
        "_id": "wave",
        "audio_name": "海浪",
        "language": "zh",
        "created_at": "2026-08-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "w", "code": "wave", "name": "平稳海浪", "en_name": "Gentle Ocean Wave"}],
    }
    fire = {
        "_id": "fire",
        "audio_name": "壁炉",
        "language": "zh",
        "created_at": "2026-07-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "f", "code": "fireplace", "name": "篝火/壁炉", "en_name": "Fireplace"}],
    }
    thunder = {
        "_id": "thunder",
        "audio_name": "雷声",
        "language": "zh",
        "created_at": "2026-06-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "t", "code": "thunder", "name": "雷声", "en_name": "Thunder"}],
    }
    insects = {
        "_id": "insects",
        "audio_name": "虫鸣",
        "language": "zh",
        "created_at": "2026-05-01T00:00:00Z",
        "content_form_tags": [
            {"tag_id": "i", "code": "crickets", "name": "蟋蟀/虫鸣", "en_name": "Crickets and Insects"},
        ],
    }
    encoder_en = MagicMock()
    encoder_en.is_loaded = True

    async def _encode_en(text: str) -> list[float]:
        return [1.0, 0.0] if text.casefold() == "thunder" else [0.0, 1.0]

    encoder_en.encode_one = AsyncMock(side_effect=_encode_en)
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "t",
                "label": "雷声",
                "name_en": "Thunder",
                "dimension": "content_form",
                "vector": [0.0, 1.0],
            }
        ]
    )
    svc = _service(
        _mongo_collection([insects, thunder, fire, wave]),
        es_search=es_search,
        encoder_en=encoder_en,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=["Natural Sound", "Gentle Ocean Wave"],
        dislike_text="Keep out thunder, barking, and sirens.",
    )

    assert [item["id"] for item in payload["list"]] == ["wave", "fire", "insects", "thunder"]


@pytest.mark.asyncio
async def test_get_audio_free_text_drops_scores_below_threshold() -> None:
    fire = {
        "_id": "fire",
        "audio_name": "壁炉",
        "language": "zh",
        "created_at": "2026-07-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "f", "code": "fireplace", "name": "篝火/壁炉", "en_name": "Fireplace"}],
    }
    thunder = {
        "_id": "thunder",
        "audio_name": "雷声",
        "language": "zh",
        "created_at": "2026-06-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "t", "code": "thunder", "name": "雷声", "en_name": "Thunder"}],
    }
    encoder_en = MagicMock()
    encoder_en.is_loaded = True

    async def _encode_en(text: str) -> list[float]:
        folded = text.casefold()
        if folded == "thunder":
            return [0.0, 1.0]
        if folded == "campfire":
            return [0.6, 0.8]
        if folded == "fireplace":
            return [1.0, 0.0]
        return [0.0, 0.0]

    encoder_en.encode_one = AsyncMock(side_effect=_encode_en)
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "f",
                "label": "篝火/壁炉",
                "name_en": "Fireplace",
                "dimension": "content_form",
                "vector": [0.0, 1.0],
            },
            {
                "id": "t",
                "label": "雷声",
                "name_en": "Thunder",
                "dimension": "content_form",
                "vector": [1.0, 0.0],
            },
        ]
    )
    svc = _service(
        _mongo_collection([fire, thunder]),
        es_search=es_search,
        encoder_en=encoder_en,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        dislike_text="campfire thunder",
    )

    assert [item["id"] for item in payload["list"]] == ["fire", "thunder"]


@pytest.mark.asyncio
async def test_get_audio_nonsense_dislike_does_not_reorder() -> None:
    svc = _service(_mongo_collection([_RAIN, _MUSIC]))

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        dislike_text="对哦温度计噢噢啊减速电机阿就死哦就",
    )

    assert [item["id"] for item in payload["list"]] == [str(_RAIN["_id"]), str(_MUSIC["_id"])]


@pytest.mark.asyncio
async def test_get_audio_query_text_ignores_quiz_preference() -> None:
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[0.1])
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.list_content_tag_vectors = AsyncMock(return_value=[])
    svc = _service(_mongo_collection([_RAIN, _MUSIC]), es_search=es_search, encoder=encoder)

    await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
        content_form_codes=["自然声"],
        preference_text="海浪拍岸",
        dislike_text="雨声",
    )

    es_search.search_by_description_vector.assert_not_called()
