"""量产音频目录查询。"""

from __future__ import annotations

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
        get_audio_root_tag_sim_threshold=0.85,
        somni_audio_catalog_cache_ttl_sec=cache_ttl_sec,
    )
    return AudioCatalogService(
        client,
        settings,
        es_search=es_search,
        encoder=encoder,
    )


def _mongo_collection(docs: list) -> MagicMock:
    collection = MagicMock()
    collection.count_documents = AsyncMock(return_value=len(docs))
    collection.find = MagicMock(return_value=_Cursor(docs))
    return collection


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
    assert [item["id"] for item in payload["materials"]] == ["a1"]
    assert payload["page"]["total"] == 1
    assert "embedding" not in payload["materials"][0]


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

    es_search.list_audio_catalog_docs.assert_awaited_once_with(size=51)
    assert [item["id"] for item in payload["materials"]] == ["a1"]


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
    es_search.list_audio_catalog_docs.assert_awaited_once_with(size=51)
    assert [item["id"] for item in payload["materials"]] == ["a1"]


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
                    "type": "content_form",
                    "code": "natural_sound",
                    "name": "自然声",
                    "name_en": "Natural Sound",
                }
            ]
        )
    )
    svc = _service(collection)
    payload = await svc.get_audio_tag()
    assert payload["tags"][0]["code"] == "natural_sound"
