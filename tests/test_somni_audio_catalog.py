"""量产音频目录查询。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AppError
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

# GetAudio 出参 tag 按 code 查词典；测试默认注入常用 content_form 条目。
_TAG_DICTIONARY = [
    {
        "_id": "dict-natural",
        "type": "content_form",
        "code": "natural_sound",
        "name": "自然声",
        "name_en": "Natural Sound",
        "status": "启用",
    },
    {
        "_id": "dict-rain",
        "type": "content_form",
        "code": "steady_rain",
        "name": "中雨/稳定雨声",
        "name_en": "Steady Rain",
        "status": "启用",
    },
    {
        "_id": "dict-music",
        "type": "content_form",
        "code": "music",
        "name": "音乐",
        "name_en": "Music",
        "status": "启用",
    },
]


class _CatalogRedis:
    """按 ex 秒过期的内存 Redis，时钟由测试推进。"""

    def __init__(self) -> None:
        self.store: dict[str, tuple[str, float | None]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.now = 0.0

    async def get(self, key: str) -> str | None:
        item = self.store.get(key)
        if item is None:
            return None
        value, deadline = item
        if deadline is not None and self.now >= deadline:
            del self.store[key]
            return None
        return value

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        deadline = None if ex is None else self.now + ex
        self.store[key] = (value, deadline)

    async def scan_iter(self, *, match: str, count: int):
        _ = count
        prefix = match.removesuffix("*")
        for key in list(self.store):
            if key.startswith(prefix):
                yield key

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self.store.pop(key, None) is not None:
                removed += 1
        return removed

    async def zrevrange(self, key: str, start: int, end: int, withscores: bool = False):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda item: (-item[1], item[0]))
        if end == -1:
            sliced = items[start:]
        else:
            sliced = items[start : end + 1]
        if withscores:
            return sliced
        return [member for member, _score in sliced]

    async def aclose(self) -> None:
        return None


def _tag_dictionary_collection(docs: list | None = None) -> MagicMock:
    rows = list(_TAG_DICTIONARY if docs is None else docs)

    def _match(query: dict) -> list:
        matched = list(rows)
        if query.get("type") is not None:
            matched = [doc for doc in matched if doc.get("type") == query["type"]]
        if query.get("status") is not None:
            matched = [doc for doc in matched if doc.get("status") == query["status"]]
        codes = (query.get("code") or {}).get("$in") if isinstance(query.get("code"), dict) else None
        if codes is not None:
            wanted = {str(code) for code in codes}
            matched = [doc for doc in matched if str(doc.get("code") or "") in wanted]
        return matched

    collection = MagicMock()
    collection.count_documents = AsyncMock(side_effect=lambda query: len(_match(query)))
    collection.find = MagicMock(side_effect=lambda query, *_a, **_k: _Cursor(_match(query)))
    return collection


def _service(
    collection: MagicMock,
    *,
    tags_collection: MagicMock | None = None,
    es_search: MagicMock | None = None,
    encoder: MagicMock | None = None,
    hot: MagicMock | None = None,
    fetch_all_hard_limit: int = 50,
    cache_ttl_sec: float = 60.0,
    redis: _CatalogRedis | None = None,
) -> AudioCatalogService:
    tags = tags_collection if tags_collection is not None else _tag_dictionary_collection()
    collections = {
        "somni_audio_materials": collection,
        "somni_audio_tag_dictionary": tags,
    }
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=lambda name: collections.get(name, collection))
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)
    settings = Settings(
        somni_mongo_db="Somni",
        somni_mongo_materials_collection="somni_audio_materials",
        somni_mongo_tag_dictionary_collection="somni_audio_tag_dictionary",
        default_page_size=1,
        max_page_size=200,
        fetch_all_hard_limit=fetch_all_hard_limit,
        get_audio_root_tag_sim_threshold=0.75,
        somni_audio_catalog_cache_ttl_sec=cache_ttl_sec,
    )
    if es_search is None:
        es_search = _es_catalog_from_collection(collection)
    else:
        if not isinstance(getattr(es_search, "list_audio_catalog_docs", None), AsyncMock):
            es_search.list_audio_catalog_docs = _es_catalog_from_collection(
                collection
            ).list_audio_catalog_docs
        if not isinstance(getattr(es_search, "list_locale_displays", None), AsyncMock):
            # 与目录夹具同源合成语言表，避免 hybrid 单测默认空 locale 把可播过滤成空列表。
            helper = _es_catalog_from_collection(collection)
            es_search.list_locale_displays = helper.list_locale_displays
        if not isinstance(getattr(es_search, "list_locale_languages", None), AsyncMock):
            helper = _es_catalog_from_collection(collection)
            es_search.list_locale_languages = helper.list_locale_languages
        if not isinstance(getattr(es_search, "search_get_audio_bm25", None), AsyncMock):
            es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
        if not isinstance(getattr(es_search, "search_get_audio_knn", None), AsyncMock):
            es_search.search_get_audio_knn = AsyncMock(return_value=[])
        if not isinstance(getattr(es_search, "search_get_audio_vector_all", None), AsyncMock):
            es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
        if not isinstance(getattr(es_search, "audio_index_exists", None), AsyncMock):
            es_search.audio_index_exists = AsyncMock(return_value=True)
        if not isinstance(
            getattr(es_search, "locale_audio_urls_by_material_ids", None), AsyncMock
        ):
            es_search.locale_audio_urls_by_material_ids = AsyncMock(
                side_effect=lambda material_ids, *, language="zh": {
                    str(doc_id): f"https://cdn.example/{doc_id}.mp3"
                    for doc_id in material_ids
                    if doc_id
                }
            )
        if not isinstance(
            getattr(es_search, "locale_displays_by_material_ids", None), AsyncMock
        ):
            es_search.locale_displays_by_material_ids = AsyncMock(
                side_effect=lambda material_ids, *, language="zh": {
                    str(doc_id): {
                        "material_id": str(doc_id),
                        "language": language,
                        "audio_name": "",
                        "description": "",
                        "summary": "",
                        "audio_url": f"https://cdn.example/{doc_id}.mp3",
                        "cover_url": "",
                    }
                    for doc_id in material_ids
                    if doc_id
                }
            )
    return AudioCatalogService(
        client,
        settings,
        es_search=es_search,
        encoder=encoder,
        hot=hot,
        redis=redis if redis is not None else _CatalogRedis(),
    )


def _locale_row_from_material(doc: dict, *, language: str) -> dict | None:
    """把测试夹具里旧的平行字段合成语言表行。"""
    material_id = str(doc.get("_id") or doc.get("id") or "")
    if not material_id:
        return None
    if language == "en":
        audio_name = str(doc.get("audio_name_en") or doc.get("audio_name") or "")
        summary = str(
            doc.get("summary_en")
            or doc.get("short_description_en")
            or doc.get("summary")
            or doc.get("short_description")
            or ""
        )
        audio_url = str(doc.get("audio_url_en") or doc.get("audio_url") or "").strip()
        cover_url = str(doc.get("cover_url_en") or doc.get("cover_url") or "").strip()
        description = str(doc.get("description_en") or doc.get("description") or "")
    else:
        # zh / fr / …：非英请求读主字段；夹具未分语言时复用主字段
        audio_name = str(doc.get("audio_name") or "")
        summary = str(doc.get("summary") or doc.get("short_description") or "")
        audio_url = str(doc.get("audio_url") or "").strip()
        cover_url = str(doc.get("cover_url") or "").strip()
        description = str(doc.get("description") or "")
    # 测试夹具常省略 url；补默认地址，与「有文案即可播」的旧夹具行为对齐。
    if not audio_url and (audio_name or summary or cover_url):
        audio_url = f"https://cdn.example/{material_id}.mp3"
    if not (audio_name or summary or audio_url or cover_url):
        return None
    return {
        "material_id": material_id,
        "language": language,
        "audio_name": audio_name,
        "summary": summary,
        "description": description,
        "audio_url": audio_url,
        "cover_url": cover_url,
    }


def _es_catalog_from_collection(collection: MagicMock) -> MagicMock:
    """测试夹具里的目录文档经 ES 列表接口返回，和 GetAudio 的读路径一致。"""
    es_search = MagicMock()

    async def _all_docs() -> list:
        try:
            cursor = collection.find({"status": True})
            return [doc async for doc in cursor]
        except TypeError:
            return []

    async def _list(*, size: int, language: str = "") -> list:
        del language
        docs = await _all_docs()
        return docs[:size]

    async def _list_locale_displays(*, language: str, size: int) -> list:
        docs = await _all_docs()
        rows: list[dict] = []
        for doc in docs:
            row = _locale_row_from_material(doc, language=language)
            if row is not None:
                rows.append(row)
            if len(rows) >= size:
                break
        return rows

    async def _list_locale_languages() -> list[str]:
        docs = await _all_docs()
        langs: list[str] = []
        for doc in docs:
            if _locale_row_from_material(doc, language="zh") is not None and "zh" not in langs:
                langs.append("zh")
            if _locale_row_from_material(doc, language="en") is not None and "en" not in langs:
                langs.append("en")
            raw = str(doc.get("language") or "").strip().lower()
            if raw and raw not in langs and _locale_row_from_material(doc, language=raw) is not None:
                langs.append(raw)
        return langs or ["zh"]

    es_search.list_audio_catalog_docs = AsyncMock(side_effect=_list)
    es_search.list_locale_displays = AsyncMock(side_effect=_list_locale_displays)
    es_search.list_locale_languages = AsyncMock(side_effect=_list_locale_languages)
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_knn = AsyncMock(return_value=[])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    es_search.audio_index_exists = AsyncMock(return_value=True)

    async def _locale_urls(material_ids: list, *, language: str = "zh") -> dict:
        docs = await _all_docs()
        by_id = {
            str(doc.get("_id") or doc.get("id") or ""): doc
            for doc in docs
            if isinstance(doc, dict)
        }
        out: dict[str, str] = {}
        for doc_id in material_ids:
            mid = str(doc_id or "")
            if not mid:
                continue
            row = _locale_row_from_material(by_id.get(mid) or {}, language=language)
            out[mid] = str((row or {}).get("audio_url") or f"https://cdn.example/{mid}.mp3")
        return out

    async def _locale_displays(material_ids: list, *, language: str = "zh") -> dict:
        docs = await _all_docs()
        by_id = {
            str(doc.get("_id") or doc.get("id") or ""): doc
            for doc in docs
            if isinstance(doc, dict)
        }
        out: dict[str, dict] = {}
        for doc_id in material_ids:
            mid = str(doc_id or "")
            if not mid:
                continue
            row = _locale_row_from_material(by_id.get(mid) or {}, language=language)
            if row is None:
                row = {
                    "material_id": mid,
                    "language": language,
                    "audio_name": "",
                    "summary": "",
                    "description": "",
                    "audio_url": f"https://cdn.example/{mid}.mp3",
                    "cover_url": "",
                }
            out[mid] = row
        return out

    es_search.locale_audio_urls_by_material_ids = AsyncMock(side_effect=_locale_urls)
    es_search.locale_displays_by_material_ids = AsyncMock(side_effect=_locale_displays)
    return es_search


def _mongo_collection(docs: list) -> MagicMock:
    collection = MagicMock()

    def _match(query: dict) -> list:
        matched = list(docs)
        if query.get("status") is True:
            matched = [doc for doc in matched if doc.get("status", True) is True]
        language = query.get("language")
        if language is not None:
            matched = [doc for doc in matched if doc.get("language") == language]
        return matched

    async def _count(query: dict) -> int:
        return len(_match(query))

    def _find(query: dict, *_args, **_kwargs) -> _Cursor:
        return _Cursor(_match(query))

    collection.count_documents = AsyncMock(side_effect=_count)
    collection.find = MagicMock(side_effect=_find)
    return collection


@pytest.mark.asyncio
async def test_get_audio_list_does_not_filter_by_material_language() -> None:
    zh_doc = {**_RAIN, "_id": "zh1", "language": "zh", "status": True}
    en_doc = {
        **_MUSIC,
        "_id": "en1",
        "audio_name": "Piano",
        "language": "en",
        "status": True,
    }
    any_doc = {
        **_RAIN,
        "_id": "any1",
        "audio_name": "雨幕",
        "language": "",
        "status": True,
    }
    collection = _mongo_collection([zh_doc, en_doc, any_doc])
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

    assert {item["id"] for item in zh_payload["list"]} == {"zh1", "en1", "any1"}
    assert {item["id"] for item in en_payload["list"]} == {"zh1", "en1", "any1"}
    es = svc._es_search
    assert es.list_audio_catalog_docs.await_count == 1


@pytest.mark.asyncio
async def test_get_audio_keeps_language_caches_separate() -> None:
    zh_doc = {**_RAIN, "_id": "zh1", "language": "zh"}
    en_doc = {**_MUSIC, "_id": "en1", "language": "en"}
    collection = _mongo_collection([zh_doc, en_doc])
    svc = _service(collection)
    es = svc._es_search

    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code="", language="zh"
    )
    catalog_reads = es.list_audio_catalog_docs.await_count
    await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code="", language="en"
    )

    # 原料目录只拉一次；en 语言表可另冷加载一次
    assert es.list_audio_catalog_docs.await_count == catalog_reads


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
        "audio_name_en",
        "audio_url",
        "cover_url",
        "description",
        "description_en",
        "vip",
        "tag",
        "is_voice",
    }
    assert payload["list"][0]["vip"] == 0
    assert payload["list"][0]["description"] == ""
    assert payload["list"][0]["tag"] == ["自然声", "中雨/稳定雨声"]


@pytest.mark.asyncio
async def test_get_audio_returns_english_display_fields() -> None:
    rain = {
        **_RAIN,
        "language": "",
        "audio_name_en": "Night Rain",
        "short_description": "轻柔雨声助眠",
        "short_description_en": "Gentle rain for sleep",
    }
    svc = _service(_mongo_collection([rain]))
    zh = await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code="", language="zh"
    )
    en = await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code="", language="en"
    )
    assert zh["list"][0]["audio_name"] == "雨夜"
    assert zh["list"][0]["description"] == "轻柔雨声助眠"
    assert zh["list"][0]["audio_name_en"] == "Night Rain"
    assert zh["list"][0]["description_en"] == "Gentle rain for sleep"
    item = en["list"][0]
    assert item["audio_name"] == "Night Rain"
    assert item["description"] == "Gentle rain for sleep"
    assert item["audio_name_en"] == "Night Rain"
    assert item["description_en"] == "Gentle rain for sleep"


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
async def test_get_audio_tag_uses_english_name_when_language_en() -> None:
    doc = {
        **_RAIN,
        # 空 language 对 en/zh 请求都可见；展示名只信词典 name_en，不信原料嵌套名
        "language": "",
        "content_form_tags": [
            {
                "tag_id": "root-rain",
                "code": "natural_sound",
                "name": "原料里的中文不应返回",
                "en_name": "Material Embedded EN",
                "parent_tag_id": None,
            },
            {
                "tag_id": "child-rain",
                "code": "steady_rain",
                "name": "原料里的二级中文",
                "en_name": "Material Nested EN",
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
async def test_get_audio_tag_resolves_zh_name_from_dictionary_by_code() -> None:
    doc = {
        **_RAIN,
        "language": "",
        "content_form_tags": [
            {
                "tag_id": "child-rain",
                "code": "steady_rain",
                "name": "原料脏数据",
                "parent_tag_id": "root-rain",
            }
        ],
    }
    svc = _service(_mongo_collection([doc]))
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="zh",
    )
    assert payload["list"][0]["tag"] == ["中雨/稳定雨声"]


@pytest.mark.asyncio
async def test_get_audio_tag_empty_string_when_dictionary_name_en_missing() -> None:
    doc = {
        "_id": "a-brown",
        "audio_name": "棕噪音",
        "language": "",
        "content_form_tags": [{"tag_id": "t1", "code": "brown_noise", "name": "忽略"}],
    }
    tags = _tag_dictionary_collection(
        [
            {
                "_id": "dict-brown",
                "type": "content_form",
                "code": "brown_noise",
                "name": "棕噪音",
                "name_en": "",
                "status": "启用",
            }
        ]
    )
    svc = _service(_mongo_collection([doc]), tags_collection=tags)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="en",
    )
    assert payload["list"][0]["tag"] == [""]


@pytest.mark.asyncio
async def test_get_audio_tag_empty_string_when_code_missing_in_dictionary() -> None:
    doc = {
        "_id": "a-orphan",
        "audio_name": "孤儿标签",
        "language": "",
        "content_form_tags": [{"tag_id": "t1", "code": "not_in_dictionary", "name": "脏名"}],
    }
    svc = _service(
        _mongo_collection([doc]),
        tags_collection=_tag_dictionary_collection([]),
    )
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="zh",
    )
    assert payload["list"][0]["tag"] == [""]


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
    collection = _mongo_collection(
        [
            {
                **_RAIN,
                "audio_name_en": "Night Rain",
                "short_description_en": "Gentle rain",
                "audio_url": "https://cdn.example/a1.mp3",
            },
            {
                **_MUSIC,
                "audio_name_en": "Piano",
                "short_description_en": "Soft piano",
                "audio_url": "https://cdn.example/a2.mp3",
            },
        ]
    )
    svc = _service(collection)
    es = svc._es_search
    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="")
    catalog_reads = es.list_audio_catalog_docs.await_count
    locale_reads = es.list_locale_displays.await_count
    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="music")
    assert es.list_audio_catalog_docs.await_count == catalog_reads
    assert es.list_locale_displays.await_count == locale_reads


@pytest.mark.asyncio
async def test_get_audio_keeps_mongo_and_es_caches_separate() -> None:
    collection = _mongo_collection([_RAIN, _MUSIC])
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[_RAIN])
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_knn = AsyncMock(
        return_value=[
            {
                **_RAIN,
                "audio_url": "https://cdn.example/a1.mp3",
                "_knn_score": 0.9,
            }
        ]
    )
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    svc = _service(collection, es_search=es_search, encoder=encoder)

    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="")
    payload = await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="雨声", tag_code=""
    )

    es_search.list_audio_catalog_docs.assert_awaited_once_with(size=51)
    assert [item["id"] for item in payload["list"]] == ["a1"]


@pytest.mark.asyncio
async def test_get_audio_cache_expires() -> None:
    collection = _mongo_collection([_RAIN])
    redis = _CatalogRedis()
    svc = _service(collection, cache_ttl_sec=10.0, redis=redis)
    es = svc._es_search

    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="")
    catalog_reads = es.list_audio_catalog_docs.await_count
    locale_reads = es.list_locale_displays.await_count
    redis.now = 11.0
    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="")

    assert es.list_audio_catalog_docs.await_count == catalog_reads * 2
    assert es.list_locale_displays.await_count == locale_reads * 2


@pytest.mark.asyncio
async def test_get_audio_cache_stays_resident() -> None:
    collection = _mongo_collection([_RAIN])
    redis = _CatalogRedis()
    svc = _service(collection, cache_ttl_sec=0, redis=redis)
    es = svc._es_search

    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="")
    catalog_reads = es.list_audio_catalog_docs.await_count
    locale_reads = es.list_locale_displays.await_count
    redis.now = 100_000.0
    await svc.get_audio(page=1, page_size=10, fetch_all=False, query_text="", tag_code="")

    assert es.list_audio_catalog_docs.await_count == catalog_reads
    assert es.list_locale_displays.await_count == locale_reads
    from app.cache.somni_audio_catalog_cache import catalog_cache_key
    from app.cache.somni_audio_locale_cache import locale_cache_key

    assert catalog_cache_key(from_es=True) in redis.store
    assert locale_cache_key("zh") in redis.store


@pytest.mark.asyncio
async def test_warm_audio_catalog_overwrites_current_key_and_drops_stale_ones() -> None:
    from app.cache.somni_audio_catalog_cache import catalog_cache_key
    from app.cache.somni_audio_locale_cache import locale_cache_key

    collection = _mongo_collection([{**_RAIN, "acoustic_metrics": {"voice_presence_ratio": 0.2}}])
    redis = _CatalogRedis()
    redis.store["somni:audio:catalog:v2:old"] = ("stale", None)
    redis.store["somni:audio:locale:v1:old"] = ("stale-locale", None)
    redis.store["somni:audio:hot:v1:query:zh:2026-W45"] = ("hot", None)
    svc = _service(collection, redis=redis)

    count = await svc.warm_audio_catalog()

    current = catalog_cache_key(from_es=True)
    assert count == 1
    assert current in redis.store
    assert locale_cache_key("zh") in redis.store
    assert "somni:audio:catalog:v2:old" not in redis.store
    assert "somni:audio:locale:v1:old" not in redis.store
    assert redis.store["somni:audio:hot:v1:query:zh:2026-W45"] == ("hot", None)
    assert '"voice_presence_ratio":0.2' in redis.store[current][0]
    assert "audio_name" not in redis.store[current][0]


@pytest.mark.asyncio
async def test_warm_audio_catalog_keeps_locale_keys_when_languages_empty() -> None:
    """语言聚合为空时不得误删已有语言表缓存。"""
    from app.cache.somni_audio_locale_cache import locale_cache_key

    collection = _mongo_collection([_RAIN])
    redis = _CatalogRedis()
    redis.store[locale_cache_key("zh")] = ('{"a1":{"audio_name":"旧雨","audio_url":"https://cdn/old.mp3"}}', None)
    redis.store[locale_cache_key("en")] = ('{"a1":{"audio_name":"Old","audio_url":"https://cdn/old.mp3"}}', None)
    es_search = _es_catalog_from_collection(collection)
    es_search.list_locale_languages = AsyncMock(return_value=[])
    svc = _service(collection, es_search=es_search, redis=redis)

    await svc.warm_audio_catalog()

    assert locale_cache_key("zh") in redis.store
    assert locale_cache_key("en") in redis.store
    assert "旧雨" in redis.store[locale_cache_key("zh")][0]


@pytest.mark.asyncio
async def test_get_audio_list_drops_empty_audio_url() -> None:
    playable = {
        **_RAIN,
        "_id": "ok",
        "audio_name": "可播",
        "audio_url": "https://cdn.example/ok.mp3",
    }
    unplayable = {
        **_MUSIC,
        "_id": "bad",
        "audio_name": "无地址",
        "audio_url": "",
        "audio_name_en": "",
        "audio_url_en": "",
    }
    svc = _service(_mongo_collection([playable, unplayable]))
    # 夹具合成语言表时无 url 会补默认 CDN；这里直接 stub 语言表。
    svc._es_search.list_locale_displays = AsyncMock(
        side_effect=lambda *, language, size: [
            {
                "material_id": "ok",
                "language": language,
                "audio_name": "可播" if language != "en" else "Playable",
                "summary": "",
                "description": "",
                "audio_url": "https://cdn.example/ok.mp3",
                "cover_url": "",
            },
            {
                "material_id": "bad",
                "language": language,
                "audio_name": "无地址" if language != "en" else "No Url",
                "summary": "",
                "description": "",
                "audio_url": "",
                "cover_url": "",
            },
        ][:size]
    )
    payload = await svc.get_audio(
        page=1, page_size=10, fetch_all=False, query_text="", tag_code="", language="zh"
    )
    assert [item["id"] for item in payload["list"]] == ["ok"]
    assert payload["total"] == 1


@pytest.mark.asyncio
async def test_get_audio_hybrid_uses_hydrated_display_directly() -> None:
    """混合检索请求语言出参直接用 hydrate，不依赖语言表全量加载。"""
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.list_locale_displays = AsyncMock(return_value=[])
    es_search.list_locale_languages = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_knn = AsyncMock(
        return_value=[
            {
                **_RAIN,
                "_id": "with-url",
                "audio_name": "Hydrated Rain",
                "summary": "from hydrate",
                "audio_url": "https://cdn.example/with-url.mp3",
                "_knn_score": 0.9,
            },
            {
                **_MUSIC,
                "_id": "no-url",
                "audio_name": "No Url",
                "audio_url": "",
                "_knn_score": 0.8,
            },
        ]
    )
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    svc = _service(_mongo_collection([_RAIN]), es_search=es_search, encoder=encoder)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="rain",
        tag_code="",
        language="en",
    )
    assert [item["id"] for item in payload["list"]] == ["with-url"]
    assert payload["list"][0]["audio_name"] == "Hydrated Rain"
    assert payload["list"][0]["description"] == "from hydrate"
    assert payload["list"][0]["audio_url"] == "https://cdn.example/with-url.mp3"
    # en 请求展示与 hydrate 同源，不应再打语言表全量接口
    es_search.list_locale_displays.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_audio_catalog_fails_with_hint_when_index_missing() -> None:
    collection = _mongo_collection([_RAIN])
    redis = _CatalogRedis()
    redis.store["somni:audio:catalog:v1:oldhash:es:zh"] = ("old-catalog", None)
    redis.store["somni:audio:hot:v1:query:zh:2026-W45"] = ("hot", None)
    before = dict(redis.store)
    es_search = _es_catalog_from_collection(collection)
    es_search.audio_index = "somni_audio_materials"
    es_search.audio_index_exists = AsyncMock(return_value=False)
    svc = _service(collection, es_search=es_search, redis=redis)

    with pytest.raises(RuntimeError, match="sync_es_from_somni.py"):
        await svc.warm_audio_catalog()

    es_search.list_audio_catalog_docs.assert_not_awaited()
    assert redis.store == before


@pytest.mark.asyncio
async def test_get_audio_and_get_hot_after_catalog_key_rebuild() -> None:
    """目录键重建后，GetAudio 仍能标出人声，GetHot 继续读原来的热榜键。"""
    from app.cache.somni_audio_catalog_cache import catalog_cache_key
    from app.server.somni.audio.hot import HotTracker, hot_redis_key
    from app.server.somni.audio.rpc import AudioRpc
    from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2

    voiced = {
        **_RAIN,
        "_id": "voiced",
        "audio_name": "人声",
        "acoustic_metrics": {"voice_presence_ratio": 0.4},
    }
    silent = {
        **_MUSIC,
        "_id": "silent",
        "language": "zh",
        "acoustic_metrics": {"voice_presence_ratio": 0},
    }
    missing = {**_RAIN, "_id": "missing", "audio_name": "无指标"}
    collection = _mongo_collection([voiced, silent, missing])
    redis = _CatalogRedis()
    redis.store["somni:audio:catalog:v1:oldhash:es:zh"] = ("old-catalog", None)
    redis.store["sleep_stage_v3_index:old:unwind"] = ("stage", None)
    hot_key = hot_redis_key(Settings(), "zh", kind="query")
    redis.store[hot_key] = ("hot-marker", None)
    redis.zsets[hot_key] = {"雨声": 6.0}
    svc = _service(
        collection,
        redis=redis,
        hot=HotTracker(redis, None, Settings()),
        cache_ttl_sec=0,
    )

    await svc.warm_audio_catalog()

    rpc = AudioRpc(svc)
    request = uburnode_somni_pb2.GetAudioReq(fetch_all=True)
    first = await rpc.GetAudio(request, MagicMock())
    reads_after_first = collection.find.call_count
    second = await rpc.GetAudio(request, MagicMock())

    assert {item.id: item.is_voice for item in first.list} == {
        "voiced": True,
        "silent": False,
        "missing": False,
    }
    assert {item.id: item.is_voice for item in second.list} == {
        "voiced": True,
        "silent": False,
        "missing": False,
    }
    assert collection.find.call_count == reads_after_first
    assert "somni:audio:catalog:v1:oldhash:es:zh" not in redis.store
    assert catalog_cache_key(from_es=True) in redis.store
    assert redis.store["sleep_stage_v3_index:old:unwind"] == ("stage", None)
    assert redis.store[hot_key] == ("hot-marker", None)

    hot = await rpc.GetHot(uburnode_somni_pb2.GetHotReq(), MagicMock())
    assert [(item.keyword, item.score) for item in hot.items] == [("雨声", 6)]


@pytest.mark.asyncio
async def test_get_audio_query_text_uses_hybrid_knn() -> None:
    collection = _mongo_collection([_RAIN])
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_knn = AsyncMock(
        return_value=[
            {
                **_RAIN,
                "audio_url": "https://cdn.example/a1.mp3",
                "_knn_score": 0.9,
            }
        ]
    )
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    svc = _service(collection, es_search=es_search, encoder=encoder)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
    )
    es_search.list_audio_catalog_docs.assert_not_called()
    es_search.search_get_audio_knn.assert_awaited()
    assert [item["id"] for item in payload["list"]] == ["a1"]
    assert payload["list"][0]["audio_name"] == "雨夜"
    assert payload["list"][0]["audio_url"] == "https://cdn.example/a1.mp3"


@pytest.mark.asyncio
async def test_get_audio_query_merges_bm25_and_knn_hits() -> None:
    piano = {
        "_id": "a-piano",
        "audio_name": "雨声钢琴",
        "language": "zh",
        "audio_url": "https://cdn.example/a-piano.mp3",
        "content_form_tags": [],
        "_bm25_score": 2.0,
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[piano])
    es_search.search_get_audio_knn = AsyncMock(
        return_value=[
            {
                **_RAIN,
                "audio_url": "https://cdn.example/a1.mp3",
                "_knn_score": 0.88,
            }
        ]
    )
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    svc = _service(_mongo_collection([_RAIN]), es_search=es_search, encoder=encoder)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
    )
    assert {item["id"] for item in payload["list"]} == {"a1", "a-piano"}
    es_search.search_get_audio_bm25.assert_awaited_once()
    kwargs = es_search.search_get_audio_bm25.await_args.kwargs
    assert kwargs["language"] == "zh"
    assert kwargs["tag_code"] == ""
    assert kwargs["size"] >= 30


@pytest.mark.asyncio
async def test_get_audio_hybrid_filters_by_request_language() -> None:
    # knn 返回值模拟语言表已 hydrate 进 audio_name 的形态
    zh_hit = {
        **_RAIN,
        "_id": "zh1",
        "audio_name": "雨夜",
        "audio_url": "https://cdn.example/zh1.mp3",
        "_knn_score": 0.9,
    }
    en_hit = {
        **_MUSIC,
        "_id": "en1",
        "audio_name": "Rain Piano",
        "audio_url": "https://cdn.example/en1.mp3",
        "_knn_score": 0.89,
    }
    any_zh = {
        **_RAIN,
        "_id": "any1",
        "audio_name": "雨幕",
        "audio_url": "https://cdn.example/any1.mp3",
        "_knn_score": 0.88,
    }
    any_en = {
        **_RAIN,
        "_id": "any1",
        "audio_name": "Rain Curtain",
        "audio_url": "https://cdn.example/any1.mp3",
        "_knn_score": 0.88,
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])

    async def _knn(*_args, language: str = "zh", **_kwargs):
        if language == "en":
            return [zh_hit, en_hit, any_en]
        return [zh_hit, en_hit, any_zh]

    es_search.search_get_audio_knn = AsyncMock(side_effect=_knn)
    svc = _service(_mongo_collection([_RAIN]), es_search=es_search, encoder=encoder)

    zh_payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
        language="zh",
    )
    en_payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="rain",
        tag_code="",
        language="en",
    )

    assert [item["id"] for item in zh_payload["list"]] == ["zh1", "any1"]
    assert [item["id"] for item in en_payload["list"]] == ["en1", "any1"]


@pytest.mark.asyncio
async def test_get_audio_hybrid_routes_by_request_language_not_query_text() -> None:
    """检索字段/模型只跟请求 language，不按输入文字识别语种。"""
    zh_hit = {
        **_RAIN,
        "_id": "zh1",
        "audio_name": "",
        "audio_url": "https://cdn.example/zh1.mp3",
        "_knn_score": 0.95,
    }
    any_hit = {
        **_RAIN,
        "_id": "any1",
        "audio_name": "rain curtain 雨",
        "audio_url": "https://cdn.example/any1.mp3",
        "_knn_score": 0.94,
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_knn = AsyncMock(return_value=[zh_hit, any_hit])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    svc = _service(_mongo_collection([_RAIN]), es_search=es_search, encoder=encoder)

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="下雨声",
        tag_code="",
        language="en",
    )

    # language=en 时 hydrate 后无英文名的 zh_hit 标题为空，字面过滤掉
    assert [item["id"] for item in payload["list"]] == ["any1"]
    encoder.encode_one.assert_awaited_once()
    assert encoder.encode_one.await_args.kwargs["language"] == "en"
    assert es_search.search_get_audio_knn.await_args.kwargs["language"] == "en"
    assert es_search.search_get_audio_bm25.await_args.kwargs["language"] == "en"


@pytest.mark.asyncio
async def test_get_audio_query_text_matches_child_tag() -> None:
    """搜索词走双路检索时，knn 命中子标签对应音频。"""
    white_noise = {
        "_id": "a3",
        "audio_name": "白噪音",
        "audio_url": "https://cdn.example/a3.mp3",
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
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_knn = AsyncMock(return_value=[{**white_noise, "_knn_score": 0.91}])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    svc = _service(_mongo_collection([_RAIN, white_noise]), es_search=es_search, encoder=encoder)
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
        "audio_url": "https://cdn.example/a-rain.mp3",
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
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[])
    es_search.search_get_audio_knn = AsyncMock(return_value=[{**rain, "_knn_score": 0.9}])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
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
    assert kwargs["hit_count"] >= 0


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
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(
        return_value=[
            {
                **_RAIN,
                "audio_url": "https://cdn.example/a1.mp3",
                "_bm25_score": 1.0,
            }
        ]
    )
    es_search.search_get_audio_knn = AsyncMock(return_value=[])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
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
        "雨声",
        language="zh",
        hit_count=1,
        tag_codes=set(),
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
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(
        return_value=[
            {
                **_RAIN,
                "audio_url": "https://cdn.example/a1.mp3",
                "_bm25_score": 1.0,
            }
        ]
    )
    es_search.search_get_audio_knn = AsyncMock(return_value=[])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
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
async def test_get_audio_query_text_without_encoder_degrades_to_bm25() -> None:
    es_search = MagicMock()
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    es_search.search_get_audio_bm25 = AsyncMock(
        return_value=[
            {
                **_RAIN,
                "audio_url": "https://cdn.example/a1.mp3",
                "_bm25_score": 1.2,
            }
        ]
    )
    es_search.search_get_audio_knn = AsyncMock(return_value=[])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    svc = _service(_mongo_collection([_RAIN]), es_search=es_search, encoder=None)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
    )
    assert [item["id"] for item in payload["list"]] == ["a1"]
    assert "vector" in payload.get("degraded", [])


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
                    "parent_id": None,
                    "status": "启用",
                }
            ]
        )
    )
    svc = _service(_mongo_collection([]), tags_collection=collection)
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
            "language": 1,
            "name": 1,
            "name_en": 1,
            "parent_id": 1,
            "status": 1,
        },
    )


def test_root_tag_query_only_content_form() -> None:
    query = catalog._root_tag_query()
    assert query["type"] == "content_form"


def test_tag_vector_for_language_prefers_en_then_fallback() -> None:
    tag = {"vector": [1.0], "vector_en": [2.0]}
    assert catalog._tag_vector_for_language(tag, "en") == [2.0]
    assert catalog._tag_vector_for_language(tag, "zh") == [1.0]
    assert catalog._tag_vector_for_language({"vector": [1.0]}, "en") == [1.0]


@pytest.mark.asyncio
async def test_get_audio_query_text_rain_matches_via_code_token() -> None:
    rain = {
        "_id": "a-rain",
        "audio_name": "听涧",
        "description": "雨声助眠",
        "audio_url": "https://cdn.example/a-rain.mp3",
        "language": "zh",
        "content_form_tags": [{"tag_id": "t1", "code": "rain", "name": "雨声"}],
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.search_get_audio_bm25 = AsyncMock(return_value=[{**rain, "_bm25_score": 3.0}])
    es_search.search_get_audio_knn = AsyncMock(return_value=[])
    es_search.search_get_audio_vector_all = AsyncMock(return_value=[])
    es_search.list_audio_catalog_docs = AsyncMock(return_value=[])
    svc = _service(_mongo_collection([rain]), es_search=es_search, encoder=encoder)
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="雨声",
        tag_code="",
        language="zh",
    )
    assert [item["id"] for item in payload["list"]] == ["a-rain"]


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (None, False),
        ({}, False),
        ({"voice_presence_ratio": None}, False),
        ({"voice_presence_ratio": 0}, False),
        ({"voice_presence_ratio": 0.0}, False),
        ({"voice_presence_ratio": 0.01}, True),
        ({"voice_presence_ratio": 1}, True),
    ],
)
def test_is_voice_follows_voice_presence_ratio(metrics, expected) -> None:
    assert catalog._is_voice(metrics) is expected


@pytest.mark.asyncio
async def test_get_audio_sets_is_voice_from_acoustic_metrics() -> None:
    silent = {
        **_RAIN,
        "_id": "silent",
        "acoustic_metrics": {"voice_presence_ratio": 0},
    }
    voiced = {
        **_MUSIC,
        "_id": "voiced",
        "acoustic_metrics": {"voice_presence_ratio": 0.4},
    }
    missing = {**_RAIN, "_id": "missing", "audio_name": "无指标"}
    svc = _service(_mongo_collection([silent, voiced, missing]))
    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=True,
        query_text="",
        tag_code="",
    )
    by_id = {item["id"]: item["is_voice"] for item in payload["list"]}
    assert by_id == {"silent": False, "voiced": True, "missing": False}
    assert "acoustic_metrics" not in payload["list"][0]


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
    hot.record_search.assert_awaited_once_with("雨声", language="zh", hit_count=1, tag_codes=set())


@pytest.mark.asyncio
async def test_prescription_search_resolves_codes_and_uses_ranked_search_without_hot(
    monkeypatch,
) -> None:
    retrieval = MagicMock()
    retrieval.search = AsyncMock(
        return_value=[
            {
                "_id": "a1",
                "status": True,
                "audio_url": "https://cdn/a.mp3",
                "_score": 0.9,
                "sleep_stage_tags": [{"code": "unwind"}],
            }
        ]
    )
    retrieval.search_prescription = retrieval.search
    service = _service(MagicMock(), hot=MagicMock())
    service._retrieval = retrieval

    materials = MagicMock()
    materials.find.return_value = _Cursor(
        [{**retrieval.search.return_value[0], "_id": "a1", "status": True}]
    )
    monkeypatch.setattr(service, "_materials", lambda: materials)

    class Cursor:
        def __init__(self, query, projection):
            self.query = query
            document = {
                "code": "breathing_guidance",
                "type": "content_form",
                "language": "zh",
                "name": "呼吸引导",
            }
            self.document = {key: value for key, value in document.items() if projection.get(key)}

        def __aiter__(self):
            self.items = iter([self.document])
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    tags = MagicMock()
    seen_queries: list[dict] = []

    def _find(query, projection):
        seen_queries.append(query)
        return Cursor(query, projection)

    tags.find.side_effect = _find
    monkeypatch.setattr(service, "_tags", lambda: tags)
    result = await service.search_prescription_audio(
        {
            "query_text": "gentle breathing",
            "sleep_stage_tags": ["unwind"],
            "content_tag_codes": ["breathing_guidance"],
            "content_tags": ["stream"],
            "disliked_tags": ["heavy_rain"],
            "top_k": 5,
            "language": "zh",
        }
    )
    assert result[0]["_id"] == "a1"
    assert seen_queries[0]["language"] == "zh"
    request = retrieval.search.await_args.args[0]
    assert request.sleep_stage_tags == ["unwind"]
    assert request.content_tags == ["breathing_guidance", "stream"]
    assert request.disliked_tags == ["heavy_rain"]
    assert request.language == "zh"
    service._hot.record_search.assert_not_called()


@pytest.mark.asyncio
async def test_prescription_fills_locale_text_not_only_url(monkeypatch) -> None:
    """原料表去掉文案后，处方候选须按请求语言补齐标题/描述/封面，不能只改 audio_url。"""
    retrieval = MagicMock()
    retrieval.search = AsyncMock(
        return_value=[
            {
                "_id": "a1",
                "status": True,
                "audio_name": "",
                "description": "",
                "cover_url": "",
                "audio_url": "",
                "sleep_stage_tags": [{"code": "unwind"}],
                "content_form_tags": [{"code": "natural_sound"}],
            }
        ]
    )
    retrieval.search_prescription = retrieval.search
    service = _service(MagicMock())
    service._retrieval = retrieval
    service._es_search.locale_displays_by_material_ids = AsyncMock(
        return_value={
            "a1": {
                "material_id": "a1",
                "language": "en",
                "audio_name": "Night Rain",
                "description": "Soft rain",
                "summary": "Soft",
                "audio_url": "https://cdn/en.mp3",
                "cover_url": "https://cdn/en.png",
            }
        }
    )

    class Cursor:
        def __init__(self, query, projection):
            self.document = {
                "code": "natural_sound",
                "type": "content_form",
                "language": "en",
                "name": "Natural",
            }

        def __aiter__(self):
            self.items = iter([self.document])
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    monkeypatch.setattr(service, "_tags", lambda: MagicMock(find=MagicMock(side_effect=Cursor)))
    result = await service.search_prescription_audio(
        {
            "query_text": "rain",
            "sleep_stage_tags": ["unwind"],
            "content_tag_codes": ["natural_sound"],
            "top_k": 3,
            "language": "en",
        }
    )
    assert result[0]["audio_url"] == "https://cdn/en.mp3"
    assert result[0]["audio_name"] == "Night Rain"
    assert result[0]["description"] == "Soft rain"
    assert result[0]["cover_url"] == "https://cdn/en.png"
    service._es_search.locale_displays_by_material_ids.assert_awaited()


@pytest.mark.asyncio
async def test_prescription_content_tag_codes_scoped_by_language(monkeypatch) -> None:
    """同一 code 的 zh/en 行不应再被当成冲突；只取请求语言那一行。"""
    retrieval = MagicMock()
    retrieval.search = AsyncMock(
        return_value=[
            {
                "_id": "a1",
                "status": True,
                "audio_url": "https://cdn/a.mp3",
                "sleep_stage_tags": [{"code": "unwind"}],
            }
        ]
    )
    retrieval.search_prescription = retrieval.search
    service = _service(MagicMock())
    service._retrieval = retrieval

    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        def __aiter__(self):
            self.items = iter(self.rows)
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    tags = MagicMock()

    def _find(query, projection):
        del projection
        assert query.get("language") == "en"
        return Cursor(
            [
                {
                    "code": "steady_rain",
                    "type": "content_form",
                    "language": "en",
                    "name": "Steady Rain",
                }
            ]
        )

    tags.find.side_effect = _find
    monkeypatch.setattr(service, "_tags", lambda: tags)
    result = await service.search_prescription_audio(
        {
            "query_text": "rain",
            "sleep_stage_tags": ["unwind"],
            "content_tag_codes": ["steady_rain"],
            "top_k": 1,
            "language": "en",
        }
    )
    assert result[0]["_id"] == "a1"
    assert retrieval.search.await_args.args[0].language == "en"


@pytest.mark.asyncio
async def test_prescription_search_fails_closed_on_unknown_tag_code(monkeypatch) -> None:
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[{"id": "must-not-return"}])
    retrieval.search_prescription = retrieval.search
    service = _service(MagicMock())
    service._retrieval = retrieval

    class EmptyCursor:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    tags = MagicMock()
    tags.find.return_value = EmptyCursor()
    monkeypatch.setattr(service, "_tags", lambda: tags)
    from app.server.somni.audio.catalog import InvalidAudioQueryError

    with pytest.raises(InvalidAudioQueryError):
        await service.search_prescription_audio(
            {"query_text": "rain", "content_tag_codes": ["missing_code"]}
        )
    retrieval.search.assert_not_called()


@pytest.mark.asyncio
async def test_prescription_search_fails_closed_on_ambiguous_tag_code(monkeypatch) -> None:
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[{"id": "must-not-return"}])
    retrieval.search_prescription = retrieval.search
    service = _service(MagicMock())
    service._retrieval = retrieval

    class Cursor:
        def __aiter__(self):
            self.items = iter(
                [
                    {
                        "code": "auditory_rhythm_stimulation",
                        "type": "content_form",
                        "name": "相同名称",
                    },
                    {
                        "code": "auditory_rhythm_stimulation",
                        "type": "mechanism",
                        "name": "相同名称",
                    },
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    tags = MagicMock()
    tags.find.return_value = Cursor()
    monkeypatch.setattr(service, "_tags", lambda: tags)

    assert (
        await service.search_prescription_audio(
            {"query_text": "rain", "content_tag_codes": ["auditory_rhythm_stimulation"]}
        )
        == []
    )
    retrieval.search.assert_not_called()


@pytest.mark.asyncio
async def test_get_audio_orders_quiz_preference_by_created_at_then_rest() -> None:
    older_natural = {
        **_RAIN,
        "_id": "old-rain",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-12-01T00:00:00Z",
    }
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

    async def _encode_zh(text: str, *, language: str = "zh") -> list[float]:
        del language
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
    second = {
        **_RAIN,
        "_id": "second",
        "audio_name": "同样新",
        "created_at": "2026-01-01T00:00:00Z",
    }
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
async def test_get_audio_skips_unknown_preference_name() -> None:
    rain = {**_RAIN, "_id": "rain", "created_at": "2026-01-01T00:00:00Z"}
    music = {**_MUSIC, "_id": "music", "created_at": "2026-06-01T00:00:00Z"}
    svc = _service(_mongo_collection([music, rain]))

    payload = await svc.get_audio(
        page=1,
        page_size=20,
        fetch_all=False,
        query_text="",
        tag_code="",
        content_form_codes=["自然声", "颜色噪音"],
        dislike_text="狗",
    )

    assert [item["id"] for item in payload["list"]] == ["rain", "music"]


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

    assert [item["id"] for item in payload["list"]] == ["alias", "voice", "rain"]


@pytest.mark.asyncio
async def test_get_audio_dislike_text_moves_matching_tags_to_end() -> None:
    steady = {
        "_id": "steady",
        "audio_name": "风扇",
        "language": "zh",
        "created_at": "2026-07-01T00:00:00Z",
        "content_form_tags": [{"tag_id": "s", "code": "steady_background", "name": "稳态背景音"}],
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

    async def _encode_zh(text: str, *, language: str = "zh") -> list[float]:
        del language
        return {
            "雨声": [1.0, 0.0, 0.0],
            "钢琴": [0.0, 1.0, 0.0],
        }.get(text, [0.0, 0.0, 1.0])

    encoder.encode_one = AsyncMock(side_effect=_encode_zh)
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "r",
                "label": "中雨/稳定雨声",
                "dimension": "content_form",
                "vector": [1.0, 0.0, 0.0],
            },
            {"id": "p", "label": "慢钢琴", "dimension": "content_form", "vector": [0.0, 1.0, 0.0]},
        ]
    )
    svc = _service(
        _mongo_collection([other, rain, piano, steady]), es_search=es_search, encoder=encoder
    )

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
        "language": "en",
        "created_at": "2026-08-01T00:00:00Z",
        "content_form_tags": [
            {
                "tag_id": "w",
                "code": "gentle_ocean_wave",
                "name": "平稳海浪",
                "en_name": "Gentle Ocean Wave",
            }
        ],
    }
    fire = {
        "_id": "fire",
        "audio_name": "壁炉",
        "language": "en",
        "created_at": "2026-07-01T00:00:00Z",
        "content_form_tags": [
            {"tag_id": "f", "code": "fireplace", "name": "篝火/壁炉", "en_name": "Fireplace"}
        ],
    }
    thunder = {
        "_id": "thunder",
        "audio_name": "雷声",
        "language": "en",
        "created_at": "2026-06-01T00:00:00Z",
        "content_form_tags": [
            {"tag_id": "t", "code": "thunder", "name": "雷声", "en_name": "Thunder"}
        ],
    }
    insects = {
        "_id": "insects",
        "audio_name": "虫鸣",
        "language": "en",
        "created_at": "2026-05-01T00:00:00Z",
        "content_form_tags": [
            {
                "tag_id": "i",
                "code": "crickets",
                "name": "蟋蟀/虫鸣",
                "en_name": "Crickets and Insects",
            },
        ],
    }
    encoder = MagicMock()
    encoder.is_loaded = True

    async def _encode(text: str, *, language: str = "zh") -> list[float]:
        del language
        return [1.0, 0.0] if text.casefold() == "thunder" else [0.0, 1.0]

    encoder.encode_one = AsyncMock(side_effect=_encode)
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "t",
                "label": "雷声",
                "name_en": "Thunder",
                "dimension": "content_form",
                "vector": [0.0, 1.0],
                "vector_en": [1.0, 0.0],
            }
        ]
    )
    svc = _service(
        _mongo_collection([insects, thunder, fire, wave]),
        es_search=es_search,
        encoder=encoder,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="en",
        content_form_codes=["Natural Sound", "Gentle Ocean Wave"],
        dislike_text="Keep out thunder, barking, and sirens.",
    )

    assert [item["id"] for item in payload["list"]] == ["wave", "fire", "insects", "thunder"]


@pytest.mark.asyncio
async def test_get_audio_free_text_drops_scores_below_threshold() -> None:
    fire = {
        "_id": "fire",
        "audio_name": "壁炉",
        "language": "en",
        "created_at": "2026-07-01T00:00:00Z",
        "content_form_tags": [
            {"tag_id": "f", "code": "fireplace", "name": "篝火/壁炉", "en_name": "Fireplace"}
        ],
    }
    thunder = {
        "_id": "thunder",
        "audio_name": "雷声",
        "language": "en",
        "created_at": "2026-06-01T00:00:00Z",
        "content_form_tags": [
            {"tag_id": "t", "code": "thunder", "name": "雷声", "en_name": "Thunder"}
        ],
    }
    encoder = MagicMock()
    encoder.is_loaded = True

    async def _encode(text: str, *, language: str = "zh") -> list[float]:
        del language
        folded = text.casefold()
        if folded == "thunder":
            return [0.0, 1.0]
        if folded == "campfire":
            return [0.6, 0.8]
        if folded == "fireplace":
            return [1.0, 0.0]
        return [0.0, 0.0]

    encoder.encode_one = AsyncMock(side_effect=_encode)
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "f",
                "label": "篝火/壁炉",
                "name_en": "Fireplace",
                "dimension": "content_form",
                "vector": [0.0, 1.0],
                "vector_en": [1.0, 0.0],
            },
            {
                "id": "t",
                "label": "雷声",
                "name_en": "Thunder",
                "dimension": "content_form",
                "vector": [1.0, 0.0],
                "vector_en": [0.0, 1.0],
            },
        ]
    )
    svc = _service(
        _mongo_collection([fire, thunder]),
        es_search=es_search,
        encoder=encoder,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="en",
        dislike_text="campfire thunder",
    )

    assert [item["id"] for item in payload["list"]] == ["fire", "thunder"]


@pytest.mark.asyncio
async def test_get_audio_zh_ignores_english_dislike_words() -> None:
    """zh 环境只走中文分词，英文 dislike 词不参与沉底。"""
    fire = {
        "_id": "fire",
        "audio_name": "壁炉",
        "language": "zh",
        "created_at": "2026-07-01T00:00:00Z",
        "content_form_tags": [
            {"tag_id": "f", "code": "fireplace", "name": "篝火/壁炉", "en_name": "Fireplace"}
        ],
    }
    thunder = {
        "_id": "thunder",
        "audio_name": "雷声",
        "language": "zh",
        "created_at": "2026-06-01T00:00:00Z",
        "content_form_tags": [
            {"tag_id": "t", "code": "thunder", "name": "雷声", "en_name": "Thunder"}
        ],
    }
    encoder = MagicMock()
    encoder.is_loaded = True
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    es_search = MagicMock()
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=[
            {
                "id": "t",
                "label": "雷声",
                "name_en": "Thunder",
                "dimension": "content_form",
                "vector": [0.0, 1.0],
                "vector_en": [1.0, 0.0],
            }
        ]
    )
    svc = _service(
        _mongo_collection([fire, thunder]),
        es_search=es_search,
        encoder=encoder,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        language="zh",
        dislike_text="Keep out thunder, barking, and sirens.",
    )

    assert [item["id"] for item in payload["list"]] == ["fire", "thunder"]
    encoder.encode_one.assert_not_awaited()


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


def _dislike_audio(audio_id: str, name: str, tag_id: str, tag_name: str, created: str) -> dict:
    return {
        "_id": audio_id,
        "audio_name": name,
        "language": "zh",
        "created_at": created,
        "description": "全程未检测到海浪声与人声。",
        "content_form_tags": [{"tag_id": tag_id, "code": tag_id, "name": tag_name}],
    }


def _description_hits(scores: dict[str, float]) -> list[dict]:
    """ES cosine 的 _score 为 (1 + cos) / 2。"""
    return [
        {"_id": audio_id, "_description_score": (1 + cos) / 2} for audio_id, cos in scores.items()
    ]


_WAVE_TAGS = [
    {
        "id": "ocean",
        "display_name": "海洋/海浪声",
        "dimension": "content_form",
        "vector": [1.0, 0.0],
    },
    {
        "id": "gentle",
        "display_name": "平稳海浪",
        "dimension": "content_form",
        "vector": [0.6, 0.8],
    },
]


def _dislike_service(
    audios: list[dict],
    scores: dict[str, float],
    *,
    tags: list[dict] | None = None,
    vectors: dict[str, list[float]] | None = None,
) -> tuple:
    word_vectors = vectors or {"海浪声": [1.0, 0.0]}
    encoder = MagicMock()
    encoder.is_loaded = True

    async def _encode(text: str, *, language: str = "zh") -> list[float]:
        del language
        return word_vectors.get(text, [0.0, 1.0])

    encoder.encode_one = AsyncMock(side_effect=_encode)
    collection = _mongo_collection(audios)
    es_search = _es_catalog_from_collection(collection)
    es_search.list_content_tag_vectors = AsyncMock(
        return_value=_WAVE_TAGS if tags is None else tags
    )
    es_search.search_by_description_vector = AsyncMock(return_value=_description_hits(scores))
    svc = _service(collection, es_search=es_search, encoder=encoder)
    return svc, es_search


def _tagged_audio(audio_id: str, name: str, created: str, *tags: tuple[str, str]) -> dict:
    return {
        "_id": audio_id,
        "audio_name": name,
        "language": "zh",
        "created_at": created,
        "content_form_tags": [
            {"tag_id": tag_id, "code": tag_id, "name": tag_name} for tag_id, tag_name in tags
        ],
    }


@pytest.mark.asyncio
async def test_get_audio_dislike_description_vector_catches_sibling_tag() -> None:
    audios = [
        _dislike_audio("gentle-wave", "康沃尔海岸的轻柔浪声", "gentle", "平稳海浪", "2026-08-01"),
        _dislike_audio("rain", "雨落林间", "rain", "物体表面雨声", "2026-07-01"),
        _dislike_audio("ocean", "明德卢夜色里的潮声", "ocean", "海洋/海浪声", "2026-06-01"),
        _dislike_audio("piano", "静夜钢琴", "piano", "慢钢琴", "2026-05-01"),
    ]
    svc, es_search = _dislike_service(
        audios,
        {"ocean": 0.70, "gentle-wave": 0.68, "rain": 0.55, "piano": 0.30},
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        dislike_text="我不喜欢海浪声",
    )

    assert [item["id"] for item in payload["list"]] == ["rain", "piano", "gentle-wave", "ocean"]
    kwargs = es_search.search_by_description_vector.await_args.kwargs
    assert kwargs["size"] == 4
    assert kwargs["language"] == "zh"
    assert kwargs["include_source"] is False


@pytest.mark.asyncio
async def test_get_audio_dislike_description_vector_respects_floor_and_margin() -> None:
    audios = [
        _dislike_audio("near", "远处的浪", "other", "其他", "2026-08-01"),
        _dislike_audio("far", "雨落林间", "rain", "物体表面雨声", "2026-07-01"),
    ]
    svc, _ = _dislike_service(audios, {"near": 0.64, "far": 0.62})

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        dislike_text="海浪声",
    )

    assert [item["id"] for item in payload["list"]] == ["near", "far"]


@pytest.mark.asyncio
async def test_get_audio_dislike_literal_matches_name_but_not_description() -> None:
    audios = [
        _dislike_audio("named-wave", "海浪拍岸", "other", "其他", "2026-08-01"),
        _dislike_audio("piano", "静夜钢琴", "piano", "慢钢琴", "2026-07-01"),
    ]
    svc, _ = _dislike_service(audios, {"named-wave": 0.30, "piano": 0.30})

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        dislike_text="海浪声",
    )

    assert [item["id"] for item in payload["list"]] == ["piano", "named-wave"]


@pytest.mark.asyncio
async def test_get_audio_dislike_tag_vector_keeps_only_tags_near_top() -> None:
    """粉噪标签与「白噪音」相似度过了根阈值，但离最高分太远，不跟着挪。"""
    noise_tags = [
        {"id": "white", "display_name": "白噪音", "dimension": "content_form", "vector": [1, 0]},
        {"id": "pink", "display_name": "粉噪", "dimension": "content_form", "vector": [0.8, 0.6]},
    ]
    audios = [
        _tagged_audio("white-noise", "白噪如幕", "2026-08-01", ("white", "白噪音")),
        _tagged_audio("pink-noise", "粉噪轻铺", "2026-07-01", ("pink", "粉噪")),
        _tagged_audio("piano", "静夜钢琴", "2026-06-01", ("piano", "慢钢琴")),
    ]
    svc, _ = _dislike_service(
        audios,
        {"white-noise": 0.73, "pink-noise": 0.60, "piano": 0.30},
        tags=noise_tags,
        vectors={"白噪音": [1.0, 0.0]},
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        dislike_text="我不喜欢白噪音",
    )

    assert [item["id"] for item in payload["list"]] == ["pink-noise", "piano", "white-noise"]


@pytest.mark.asyncio
async def test_get_audio_dislike_spreads_to_child_tags_of_description_hits() -> None:
    """描述命中的音频挂着「岸边水波声」，同标签但描述没命中的也挪后；一级标签自然声不扩散。"""
    tags = [
        {
            "id": "nature",
            "display_name": "自然声",
            "dimension": "content_form",
            "parent_id": "",
            "vector": [0.9, 0.4359],
        },
        {
            "id": "ocean",
            "display_name": "海洋/海浪声",
            "dimension": "content_form",
            "parent_id": "nature",
            "vector": [1.0, 0.0],
        },
        {
            "id": "shore",
            "display_name": "岸边水波声",
            "dimension": "content_form",
            "parent_id": "nature",
            "vector": [0.7, 0.71414],
        },
    ]
    nature = ("nature", "自然声")
    shore = ("shore", "岸边水波声")
    audios = [
        _tagged_audio("gentle-b", "潮池的浪花", "2026-09-01", nature, shore),
        _tagged_audio("gentle-a", "轻柔浪声", "2026-08-01", nature, shore),
        _tagged_audio("rain", "雨落林间", "2026-07-01", nature, ("rain", "物体表面雨声")),
        _tagged_audio("ocean-a", "夜色潮声", "2026-06-01", nature, ("ocean", "海洋/海浪声")),
    ]
    svc, es_search = _dislike_service(
        audios,
        {"ocean-a": 0.70, "gentle-a": 0.68, "rain": 0.55, "gentle-b": 0.40},
        tags=tags,
    )

    payload = await svc.get_audio(
        page=1,
        page_size=10,
        fetch_all=False,
        query_text="",
        tag_code="",
        dislike_text="不喜欢海浪声",
    )

    assert [item["id"] for item in payload["list"]] == ["rain", "gentle-b", "gentle-a", "ocean-a"]
    es_search.search_by_description_vector.assert_awaited_once()


def test_order_like_neutral_dislike_three_bands_by_created_at() -> None:
    """喜欢 / 中性 / 讨厌 三段；段内从新到旧；既喜欢又讨厌归讨厌。"""
    from app.server.somni.audio.catalog import _order_like_neutral_dislike

    docs = [
        {"id": "like-old", "created_at": "2026-01-01T00:00:00Z"},
        {"id": "dislike-new", "created_at": "2026-06-01T00:00:00Z"},
        {"id": "neutral-new", "created_at": "2026-05-01T00:00:00Z"},
        {"id": "like-new", "created_at": "2026-04-01T00:00:00Z"},
        {"id": "both", "created_at": "2026-08-01T00:00:00Z"},
        {"id": "dislike-old", "created_at": "2026-02-01T00:00:00Z"},
        {"id": "neutral-old", "created_at": "2026-03-01T00:00:00Z"},
    ]
    ordered = _order_like_neutral_dislike(
        docs,
        preferred_ids={"like-old", "like-new", "both"},
        disliked_ids={"dislike-new", "dislike-old", "both"},
    )
    assert [d["id"] for d in ordered] == [
        "like-new",
        "like-old",
        "neutral-new",
        "neutral-old",
        "both",
        "dislike-new",
        "dislike-old",
    ]


def test_filter_docs_keeps_blank_and_null_language() -> None:
    from app.server.somni.audio.catalog import _filter_docs_by_request_language

    docs = [
        {"id": "1", "language": ""},
        {"id": "2", "language": None},
        {"id": "3"},
        {"id": "4", "language": "null"},
        {"id": "5", "language": "None"},
        {"id": "6", "language": "zh"},
        {"id": "7", "language": "en"},
    ]
    kept = _filter_docs_by_request_language(docs, "zh")
    assert [d["id"] for d in kept] == ["1", "2", "3", "4", "5", "6", "7"]
    kept_en = _filter_docs_by_request_language(docs, "en")
    assert [d["id"] for d in kept_en] == ["1", "2", "3", "4", "5", "6", "7"]
