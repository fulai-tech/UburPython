"""GetAudioByIds：按 id 数组从 ES 取启用详情，language 就地填展示字段。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AppError
from app.server.somni.audio.catalog import AudioCatalogService, InvalidAudioQueryError
from tests.test_somni_audio_catalog import _CatalogRedis, _tag_dictionary_collection


def _svc(
    *,
    es_docs_by_id: dict[str, dict] | None = None,
    tags_collection: MagicMock | None = None,
) -> tuple[AudioCatalogService, MagicMock]:
    es_search = MagicMock()
    store = dict(es_docs_by_id or {})

    async def _by_ids(ids: list[str]) -> list[dict]:
        out: list[dict] = []
        for audio_id in ids:
            doc = store.get(audio_id)
            if doc is None:
                continue
            if doc.get("status", True) is not True:
                continue
            out.append({**doc, "_id": audio_id})
        return out

    es_search.get_audio_docs_by_ids = AsyncMock(side_effect=_by_ids)
    es_search.list_locale_displays = AsyncMock(return_value=[])

    materials = MagicMock()
    tags = tags_collection if tags_collection is not None else _tag_dictionary_collection()
    collections = {
        "somni_audio_materials": materials,
        "somni_audio_tag_dictionary": tags,
    }
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=lambda name: collections[name])
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)
    settings = Settings(
        somni_mongo_db="Somni",
        somni_mongo_materials_collection="somni_audio_materials",
        somni_mongo_tag_dictionary_collection="somni_audio_tag_dictionary",
    )
    svc = AudioCatalogService(
        client,
        settings,
        es_search=es_search,
        redis=_CatalogRedis(),
    )
    return svc, es_search


@pytest.mark.asyncio
async def test_get_audio_by_ids_fr_uses_french_locale_not_chinese() -> None:
    """language=fr 只查法语语言表，不回退中文。"""
    svc, es = _svc(
        es_docs_by_id={
            "a1": {
                "status": True,
                "vip": 0,
                "content_form_tags": [],
                "acoustic_metrics": {"voice_presence_ratio": 0},
            }
        }
    )

    async def _locales(*, language: str, size: int) -> list[dict]:
        del size
        if language == "zh":
            return [
                {
                    "material_id": "a1",
                    "language": "zh",
                    "audio_name": "雨夜",
                    "summary": "轻柔雨声",
                    "audio_url": "https://cdn/zh.mp3",
                    "cover_url": "https://cdn/zh.png",
                }
            ]
        if language == "fr":
            return [
                {
                    "material_id": "a1",
                    "language": "fr",
                    "audio_name": "Pluie de nuit",
                    "summary": "Pluie douce",
                    "audio_url": "https://cdn/fr.mp3",
                    "cover_url": "https://cdn/fr.png",
                }
            ]
        return []

    es.list_locale_displays = AsyncMock(side_effect=_locales)
    payload = await svc.get_audio_by_ids(ids=["a1"], language="fr")
    item = payload["list"][0]
    assert item["audio_name"] == "Pluie de nuit"
    assert item["description"] == "Pluie douce"
    assert item["audio_url"] == "https://cdn/fr.mp3"
    assert item["cover_url"] == "https://cdn/fr.png"
    called_langs = [call.kwargs["language"] for call in es.list_locale_displays.await_args_list]
    assert called_langs == ["fr"]


@pytest.mark.asyncio
async def test_get_audio_by_ids_missing_fr_locale_does_not_fallback_to_zh() -> None:
    """请求 fr 但语言表无 fr 行时，不得回退显示中文。"""
    svc, es = _svc(
        es_docs_by_id={
            "a1": {
                "status": True,
                "vip": 0,
                "content_form_tags": [],
                "acoustic_metrics": {"voice_presence_ratio": 0},
            }
        }
    )

    async def _locales(*, language: str, size: int) -> list[dict]:
        del size
        if language == "zh":
            return [
                {
                    "material_id": "a1",
                    "language": "zh",
                    "audio_name": "雨夜",
                    "summary": "轻柔雨声",
                    "audio_url": "https://cdn/zh.mp3",
                    "cover_url": "https://cdn/zh.png",
                }
            ]
        return []

    es.list_locale_displays = AsyncMock(side_effect=_locales)
    payload = await svc.get_audio_by_ids(ids=["a1"], language="fr")
    item = payload["list"][0]
    assert item["audio_name"] == ""
    assert item["description"] == ""
    assert item["audio_url"] == ""
    assert item["cover_url"] == ""
    assert [call.kwargs["language"] for call in es.list_locale_displays.await_args_list] == [
        "fr"
    ]


@pytest.mark.asyncio
async def test_get_audio_by_ids_localizes_fields_and_tags() -> None:
    svc, es = _svc(
        es_docs_by_id={
            "a1": {
                "status": True,
                "audio_name": "雨夜",
                "audio_name_en": "Night Rain",
                "short_description": "轻柔雨声",
                "short_description_en": "Gentle rain",
                "audio_url": "https://cdn/a.mp3",
                "cover_url": "https://cdn/a.png",
                "vip": 0,
                "content_form_tags": [{"code": "steady_rain"}],
                "acoustic_metrics": {"voice_presence_ratio": 0},
            }
        }
    )
    zh = await svc.get_audio_by_ids(ids=["a1"], language="zh")
    en = await svc.get_audio_by_ids(ids=["a1"], language="en")

    assert zh["list"] == [
        {
            "id": "a1",
            "audio_name": "雨夜",
            "audio_url": "https://cdn/a.mp3",
            "cover_url": "https://cdn/a.png",
            "description": "轻柔雨声",
            "vip": 0,
            "tag": ["中雨/稳定雨声"],
            "is_voice": False,
        }
    ]
    assert en["list"][0]["audio_name"] == "Night Rain"
    assert en["list"][0]["description"] == "Gentle rain"
    assert en["list"][0]["tag"] == ["Steady Rain"]
    assert "audio_name_en" not in en["list"][0]
    assert "description_en" not in en["list"][0]
    assert es.get_audio_docs_by_ids.await_count == 2


@pytest.mark.asyncio
async def test_get_audio_by_ids_preserves_order_dedupes_skips_missing_and_disabled() -> None:
    svc, es = _svc(
        es_docs_by_id={
            "a": {"status": True, "audio_name": "A", "audio_url": "u", "content_form_tags": []},
            "b": {"status": False, "audio_name": "B", "audio_url": "u", "content_form_tags": []},
            "c": {"status": True, "audio_name": "C", "audio_url": "u", "content_form_tags": []},
        }
    )
    payload = await svc.get_audio_by_ids(
        ids=["", "c", "a", "c", "missing", "b", "  a  "],
        language="zh",
    )
    assert [item["id"] for item in payload["list"]] == ["c", "a"]
    es.get_audio_docs_by_ids.assert_awaited_once_with(["c", "a", "missing", "b"])


@pytest.mark.asyncio
async def test_get_audio_by_ids_empty_ids_returns_empty_without_es() -> None:
    svc, es = _svc(es_docs_by_id={"a": {"status": True, "audio_name": "A"}})
    payload = await svc.get_audio_by_ids(ids=[], language="zh")
    assert payload == {"list": []}
    es.get_audio_docs_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_get_audio_by_ids_rejects_over_limit() -> None:
    svc, _es = _svc()
    with pytest.raises(InvalidAudioQueryError):
        await svc.get_audio_by_ids(ids=[f"id-{i}" for i in range(101)], language="zh")


@pytest.mark.asyncio
async def test_get_audio_by_ids_requires_es() -> None:
    settings = Settings(somni_mongo_db="Somni")
    svc = AudioCatalogService(MagicMock(), settings, es_search=None, redis=_CatalogRedis())
    with pytest.raises(AppError) as exc:
        await svc.get_audio_by_ids(ids=["a1"], language="zh")
    assert exc.value.status_code == 503
