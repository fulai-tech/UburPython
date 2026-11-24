"""量产音频原料批量增加、更新、删除。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from bson import ObjectId
from elasticsearch import NotFoundError

from app.server.somni.audio.rpc import AudioRpc
from app.server.somni.audio.writes import (
    AUDIO_WRITE_BATCH_LIMIT,
    audio_material_input_to_fields,
    create_audios,
    delete_audios,
    update_audios,
)
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _context() -> MagicMock:
    return MagicMock()


def _locale_row(**extra: object) -> dict:
    row = {
        "_id": ObjectId(),
        "language": "zh",
        "audio_name": "雨声",
        "audio_url": "https://cdn.example/a.mp3",
        "cover_url": "",
        "description": "",
        "summary": "",
    }
    row.update(extra)
    return row


def _writer() -> SimpleNamespace:
    es = SimpleNamespace(
        audio_index="somni_audio_materials",
        material_locales_index="somni_audio_material_locales",
        refresh_audio_index=AsyncMock(),
        _client=SimpleNamespace(
            index=AsyncMock(),
            update=AsyncMock(),
            delete=AsyncMock(),
            bulk=AsyncMock(return_value={"errors": False, "items": []}),
            get=AsyncMock(
                return_value=SimpleNamespace(body={"_source": {"description_vector": [0.9, 0.1]}})
            ),
        ),
    )
    encoder = SimpleNamespace(
        encode_one=AsyncMock(return_value=[0.2, 0.4]),
        encode=AsyncMock(side_effect=lambda texts, language="zh": [[0.2, 0.4] for _ in texts]),
    )
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value={"_id": ObjectId(), "status": True})
    collection.replace_one = AsyncMock()
    collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=ObjectId()))
    collection.find = MagicMock(return_value=SimpleNamespace(to_list=AsyncMock(return_value=[])))
    locales = MagicMock()
    row = _locale_row()
    locales.find_one_and_update = AsyncMock(return_value=row)
    locales.find_one = AsyncMock(
        side_effect=lambda query: row if query.get("language") == "zh" else None
    )
    locales.delete_many = AsyncMock()
    locales.delete_one = AsyncMock()
    locales.replace_one = AsyncMock()
    locales.insert_many = AsyncMock()
    locales.find = MagicMock(return_value=SimpleNamespace(to_list=AsyncMock(return_value=[row])))
    tags = MagicMock()
    tags.find = MagicMock(return_value=SimpleNamespace(to_list=AsyncMock(return_value=[])))
    return SimpleNamespace(
        collection=collection,
        locales=locales,
        tags=tags,
        locale_row=row,
        _es_search=es,
        _encoder=encoder,
        _catalog_cache=SimpleNamespace(_redis=object()),
        _retrieval=SimpleNamespace(clear_sleep_stage_cache=AsyncMock()),
        _materials=lambda: None,
        _locales=lambda: None,
        _tags=lambda: None,
    )


def _bind(service: SimpleNamespace) -> SimpleNamespace:
    service._materials = lambda: service.collection
    service._locales = lambda: service.locales
    service._tags = lambda: service.tags
    return service


@pytest.mark.asyncio
async def test_successful_write_refreshes_es_before_clearing_cache(monkeypatch) -> None:
    """写成功后先 refresh，再清目录缓存，避免立刻读回旧结果并长期缓存。"""
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    order: list[str] = []

    async def _refresh() -> None:
        order.append("refresh")

    async def _clear(_redis: object) -> int:
        order.append("clear")
        return 1

    service._es_search.refresh_audio_index = AsyncMock(side_effect=_refresh)
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        _clear,
    )

    results = await create_audios(
        service,
        [
            {
                "audio_name": "雨声",
                "language": "zh",
                "audio_url": "https://cdn.example/a.mp3",
            }
        ],
    )

    assert results[0]["ok"] is True
    assert order == ["refresh", "clear"]


@pytest.mark.asyncio
async def test_write_skips_cache_clear_when_es_refresh_fails(monkeypatch) -> None:
    """refresh 失败时保留旧缓存，避免把未可见的旧 ES 结果重新写入无 TTL 缓存。"""
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    clear_cache = AsyncMock()
    service._es_search.refresh_audio_index = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        clear_cache,
    )

    results = await create_audios(
        service,
        [
            {
                "audio_name": "雨声",
                "language": "zh",
                "audio_url": "https://cdn.example/a.mp3",
            }
        ],
    )

    assert results[0]["ok"] is True
    clear_cache.assert_not_awaited()
    service._retrieval.clear_sleep_stage_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_audios_returns_each_item_and_indexes_chinese_vector(monkeypatch) -> None:
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    clear_cache = AsyncMock()
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        clear_cache,
    )

    results = await create_audios(
        service,
        [
            {},
            {
                "audio_name": "雨声",
                "language": "zh",
                "audio_url": "https://cdn.example/a.mp3",
                "status": False,
            },
            {"audio_name": "雨声", "audio_url": "https://cdn.example/a.mp3"},
        ],
    )

    assert results[0]["ok"] is False
    assert results[0]["message"] == "audio_name 不能为空"
    assert results[1]["ok"] is True
    assert results[1]["id"] == str(inserted)
    assert results[2]["ok"] is False
    assert results[2]["message"] == "language 不能为空"
    doc = service.collection.insert_one.await_args.args[0]
    assert doc["status"] is False
    assert "audio_name" not in doc
    assert "audio_url" not in doc
    assert "description" not in doc
    assert "summary" not in doc
    assert "cover_url" not in doc
    assert "mechanism_tags" not in doc
    assert "medical_risk_tags" not in doc
    assert "embedding" not in doc
    locale = service.locales.find_one_and_update.await_args.args[1]["$set"]
    assert locale["audio_name"] == "雨声"
    assert locale["audio_url"] == "https://cdn.example/a.mp3"
    assert locale["language"] == "zh"
    material_indexed = [
        call.kwargs for call in service._es_search._client.index.await_args_list
        if call.kwargs["index"] == "somni_audio_materials"
    ][0]
    assert material_indexed["id"] == str(inserted)
    assert "audio_name" not in material_indexed["document"]
    assert "description_vector" not in material_indexed["document"]
    assert "mechanism_tags" not in material_indexed["document"]
    locale_indexed = [
        call.kwargs for call in service._es_search._client.index.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    ][0]
    assert locale_indexed["document"]["audio_name"] == "雨声"
    # 无 content_form 标签时 description_text 为空，不写检索向量
    assert locale_indexed["document"].get("description_text", "") == ""
    assert "description_vector" not in locale_indexed["document"]
    clear_cache.assert_awaited_once()
    service._retrieval.clear_sleep_stage_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_audios_sets_only_passed_fields(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    service.locale_row["summary"] = "短"
    service.locales.find_one_and_update = AsyncMock(return_value=service.locale_row)
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [
            {"id": "not-an-id", "fields": {"summary": "不会写", "language": "zh"}},
            {"id": str(material_id), "fields": {"summary": "短", "language": "zh"}},
            {"id": str(material_id), "fields": {}},
            {"id": str(material_id), "fields": {"summary": "短"}},
        ],
    )

    assert [item["ok"] for item in results] == [False, True, False, False]
    assert results[0]["message"] == "id 不合法"
    assert results[2]["message"] == "没有要更新的字段"
    assert results[3]["message"] == "language 不能为空"
    service.collection.find_one_and_update.assert_not_called()
    payload = service.locales.find_one_and_update.await_args.args[1]["$set"]
    assert payload["summary"] == "短"
    assert "audio_name" not in payload
    # summary 不进入 description_text；同语言更新不重算向量
    service._encoder.encode_one.assert_not_awaited()
    patched = service._es_search._client.update.await_args.kwargs
    assert patched["index"] == "somni_audio_material_locales"
    assert patched["doc"]["summary"] == "短"
    assert "description_text" not in patched["doc"]


@pytest.mark.asyncio
async def test_update_clears_tags_and_drops_index_without_audio_url(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    service.locale_row["audio_url"] = ""
    service.locales.find_one_and_update = AsyncMock(return_value=service.locale_row)
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "sleep_stage_tags": [], "status": True}
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [
            {
                "id": str(material_id),
                "fields": {"language": "zh", "audio_url": "", "sleep_stage_tags": []},
            }
        ],
    )

    assert results[0]["ok"] is True
    material = service.collection.find_one_and_update.await_args.args[1]["$set"]
    assert material["sleep_stage_tags"] == []
    assert "audio_url" not in material
    locale = service.locales.find_one_and_update.await_args.args[1]["$set"]
    assert locale["audio_url"] == ""
    service._es_search._client.delete.assert_not_called()
    material_patch = next(
        call.kwargs
        for call in service._es_search._client.update.await_args_list
        if call.kwargs["index"] == "somni_audio_materials"
    )
    assert material_patch["doc"]["sleep_stage_tags"] == []
    assert "audio_url" not in material_patch["doc"]


@pytest.mark.asyncio
async def test_update_recomputes_vector_when_title_description_or_tags_are_sent(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    service.locale_row["description"] = "同样的描述"
    service.locales.find_one_and_update = AsyncMock(return_value=service.locale_row)
    service.collection.find_one_and_update = AsyncMock(
        return_value={
            "_id": material_id,
            "sleep_stage_tags": [{"name": "入睡"}],
            "status": True,
        }
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [
            {
                "id": str(material_id),
                "fields": {
                    "language": "zh",
                    "description": "同样的描述",
                    "sleep_stage_tags": [{"name": "入睡"}],
                },
            }
        ],
    )

    assert results[0]["ok"] is True
    # description 只驱动原料 embedding；语言索引 description_text 仅看 content_form
    encoded_texts = [call.args[0] for call in service._encoder.encode_one.await_args_list]
    assert encoded_texts == ["同样的描述"]
    stored = service.collection.find_one_and_update.await_args.args[1]["$set"]
    assert stored["embedding"] == [0.2, 0.4]
    locale_patch = next(
        call.kwargs
        for call in service._es_search._client.update.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    )
    # 无 content_form 时重算结果为空文本并清向量
    fields = locale_patch["script"]["params"]["fields"]
    assert fields["description"] == "同样的描述"
    assert fields["description_text"] == ""
    assert "description_vector" not in fields
    assert "embedding" not in fields


@pytest.mark.asyncio
async def test_tag_dictionary_unavailable_keeps_update_failed(monkeypatch) -> None:
    """词典 Mongo 读失败时不得把语言索引写成空文本/清向量。"""
    service = _bind(_writer())
    material_id = ObjectId()
    service.collection.find_one = AsyncMock(
        return_value={
            "_id": material_id,
            "content_form_tags": [{"code": "rain"}],
            "status": True,
        }
    )
    service.collection.find_one_and_update = AsyncMock(
        return_value={
            "_id": material_id,
            "content_form_tags": [{"code": "rain", "name": "雨"}],
            "status": True,
        }
    )
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {
                        "_id": ObjectId(),
                        "material_id": str(material_id),
                        "language": "zh",
                        "audio_name": "雨曲",
                    }
                ]
            )
        )
    )
    service.tags.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(side_effect=TimeoutError("mongo timeout"))
        )
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [
            {
                "id": str(material_id),
                "fields": {
                    "content_form_tags": [{"code": "rain", "name": "雨"}],
                },
            }
        ],
    )

    assert results[0]["ok"] is False
    assert "标签词典不可用" in results[0]["message"]
    # 失败后应补偿原料，且不应成功写出空语言检索文本
    service.collection.replace_one.assert_awaited()
    # 语言索引回滚同样会因词典不可用失败，但原料索引必须照样回滚到旧标签
    material_restores = [
        call.kwargs
        for call in service._es_search._client.index.await_args_list
        if call.kwargs["index"] == "somni_audio_materials"
    ]
    assert material_restores
    assert material_restores[-1]["document"]["content_form_tags"] == [{"code": "rain"}]
    locale_writes = [
        call.kwargs
        for call in service._es_search._client.update.await_args_list
        if call.kwargs.get("index") == "somni_audio_material_locales"
    ]
    assert not any(
        call.get("script", {}).get("params", {}).get("fields", {}).get("description_text")
        == ""
        for call in locale_writes
    )


@pytest.mark.asyncio
async def test_failed_update_deletes_index_of_converted_blank_locale(monkeypatch) -> None:
    """空语言行被本次写入赋了语言且改了标签，失败后 Mongo 恢复为空语言，ES 文档也要删掉。"""
    service = _bind(_writer())
    material_id = ObjectId()
    blank = {"_id": ObjectId(), "material_id": str(material_id), "language": "", "audio_name": "雨"}
    converted = {**blank, "language": "zh"}
    service.locales.find_one = AsyncMock(
        side_effect=lambda query: blank if query.get("language") == "" else None
    )
    service.locales.find_one_and_update = AsyncMock(return_value=dict(converted))
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[dict(blank)]))
    )
    service.collection.find_one = AsyncMock(
        return_value={"_id": material_id, "content_form_tags": [], "status": True}
    )
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "content_form_tags": [], "status": True}
    )
    service._es_search._client.update = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [
            {
                "id": str(material_id),
                "fields": {"language": "zh", "content_form_tags": []},
            }
        ],
    )

    assert results[0]["ok"] is False
    service.locales.replace_one.assert_awaited()
    deleted = [
        call.kwargs
        for call in service._es_search._client.delete.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    ]
    assert [item["id"] for item in deleted] == [str(blank["_id"])]


@pytest.mark.asyncio
async def test_reenable_rebuilds_all_locale_search_text(monkeypatch) -> None:
    """停用素材重新启用：所有语言行按当前标签重算检索文本与向量。"""
    service = _bind(_writer())
    material_id = ObjectId()
    material = {"_id": material_id, "content_form_tags": [{"code": "rain"}]}
    service.collection.find_one = AsyncMock(return_value={**material, "status": False})
    service.collection.find_one_and_update = AsyncMock(return_value={**material, "status": True})
    zh_id, en_id = ObjectId(), ObjectId()
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"_id": zh_id, "material_id": str(material_id), "language": "zh"},
                    {"_id": en_id, "material_id": str(material_id), "language": "en"},
                ]
            )
        )
    )

    def _dictionary(query, **_kwargs):
        names = {"zh": "雨声", "en": "Rain"}
        return SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"type": "content_form", "code": "rain", "language": lang, "name": names[lang]}
                    for lang in query["language"]["$in"]
                ]
            )
        )

    service.tags.find = MagicMock(side_effect=_dictionary)
    service.collection.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[material]))
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service, [{"id": str(material_id), "fields": {"status": True}}]
    )

    assert results[0]["ok"] is True
    locale_patches = {
        call.kwargs["id"]: call.kwargs["script"]["params"]["fields"]
        for call in service._es_search._client.update.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    }
    assert locale_patches[str(zh_id)]["description_text"] == "雨声"
    assert locale_patches[str(zh_id)]["description_vector"] == [0.2, 0.4]
    assert locale_patches[str(en_id)]["description_text"] == "Rain"
    assert locale_patches[str(en_id)]["description_vector_en"] == [0.2, 0.4]
    # 写后复查与 Mongo 一致，不再重写
    service._es_search._client.bulk.assert_not_awaited()


@pytest.mark.asyncio
async def test_tag_rename_after_update_write_is_fixed_by_verify(monkeypatch) -> None:
    """单条写入用了旧词典名，写完时标签已改名：写后复查按新名重写。"""
    service = _bind(_writer())
    material_id = ObjectId()
    material = {"_id": material_id, "content_form_tags": [{"code": "rain"}], "status": True}
    service.collection.find_one = AsyncMock(return_value=dict(material))
    service.collection.find_one_and_update = AsyncMock(return_value=dict(material))
    service.collection.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[material]))
    )
    locale_id = ObjectId()
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[{"_id": locale_id, "material_id": str(material_id), "language": "zh"}]
            )
        )
    )
    reads: list[str] = []

    def _dictionary(*_args, **_kwargs):
        name = "旧雨" if not reads else "新雨"
        reads.append(name)
        return SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"type": "content_form", "code": "rain", "language": "zh", "name": name}
                ]
            )
        )

    service.tags.find = MagicMock(side_effect=_dictionary)
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [{"id": str(material_id), "fields": {"content_form_tags": [{"code": "rain"}]}}],
    )

    assert results[0]["ok"] is True
    patched = next(
        call.kwargs["script"]["params"]["fields"]
        for call in service._es_search._client.update.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    )
    assert patched["description_text"] == "旧雨"
    service._es_search._client.bulk.assert_awaited_once()
    operations = service._es_search._client.bulk.await_args.kwargs["operations"]
    assert operations[0]["update"]["_id"] == str(locale_id)
    assert operations[1]["script"]["params"]["fields"]["description_text"] == "新雨"


@pytest.mark.asyncio
async def test_update_keeps_vector_when_only_status_changes(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "status": False}
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [{"id": str(material_id), "fields": {"status": False}}],
    )

    assert results[0]["ok"] is True
    service._encoder.encode_one.assert_not_called()
    service._es_search._client.index.assert_not_called()
    patched = service._es_search._client.update.await_args.kwargs["doc"]
    assert patched["status"] is False
    assert "description_vector" not in patched
    stored = service.collection.find_one_and_update.await_args.args[1]["$set"]
    assert "embedding" not in stored


@pytest.mark.asyncio
async def test_update_indexes_full_document_when_es_doc_is_missing(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    service._es_search._client.update = AsyncMock(side_effect=NotFoundError("missing", None, None))
    service.locale_row.update(audio_name="雨声", description="雨声描述", audio_url="https://cdn.example/a.mp3")
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "status": False}
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [{"id": str(material_id), "fields": {"status": False}}],
    )

    assert results[0]["ok"] is True
    indexed = service._es_search._client.index.await_args.kwargs["document"]
    assert indexed["status"] is False
    assert "description_vector" not in indexed
    assert "audio_name" not in indexed


@pytest.mark.asyncio
async def test_create_embeds_description_and_update_clears_it(monkeypatch) -> None:
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    material_id = ObjectId()
    service.locale_row["description"] = ""
    service.locales.find_one_and_update = AsyncMock(return_value=service.locale_row)
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "status": True}
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    created = await create_audios(
        service,
        [
            {
                "audio_name": "雨声",
                "language": "zh",
                "audio_url": "https://cdn.example/a.mp3",
                "description": "雨声描述",
            }
        ],
    )
    assert created[0]["ok"] is True
    created_doc = service.collection.insert_one.await_args.args[0]
    assert created_doc["embedding"] == [0.2, 0.4]

    cleared = await update_audios(
        service,
        [{"id": str(material_id), "fields": {"language": "zh", "description": "  "}}],
    )
    assert cleared[0]["ok"] is True
    update = service.collection.find_one_and_update.await_args.args[1]
    assert "embedding" not in update["$set"]
    assert update["$unset"] == {"embedding": ""}


@pytest.mark.asyncio
async def test_delete_audios_continues_after_missing_item(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    service.collection.delete_one = AsyncMock(
        side_effect=[SimpleNamespace(deleted_count=0), SimpleNamespace(deleted_count=1)]
    )
    clear_cache = AsyncMock()
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        clear_cache,
    )

    missing = ObjectId()
    results = await delete_audios(
        service,
        [{"id": str(missing)}, {"id": str(material_id)}],
    )

    assert results[0]["ok"] is False
    assert "不存在" in results[0]["message"]
    assert results[1]["ok"] is True
    material_deletes = [
        call
        for call in service._es_search._client.delete.await_args_list
        if call.kwargs["index"] == "somni_audio_materials"
    ]
    assert len(material_deletes) == 1
    clear_cache.assert_awaited_once()


def test_material_input_keeps_only_present_fields() -> None:
    message = uburnode_somni_pb2.AudioMaterialInput(summary="短")
    message.sleep_stage_tags.CopyFrom(uburnode_somni_pb2.AudioTagRefList())

    assert audio_material_input_to_fields(message) == {"summary": "短", "sleep_stage_tags": []}


@pytest.mark.asyncio
async def test_rpc_create_and_update_pass_partial_fields() -> None:
    service = MagicMock()
    service.create_audios = AsyncMock(
        return_value=[
            {"index": 0, "ok": True, "id": "m1", "message": ""},
            {"index": 1, "ok": False, "id": "", "message": "audio_name 不能为空"},
        ]
    )
    service.update_audios = AsyncMock(
        return_value=[{"index": 0, "ok": True, "id": "m1", "message": ""}]
    )
    rpc = AudioRpc(service)
    created = await rpc.CreateAudio(
        uburnode_somni_pb2.CreateAudioReq(
            items=[
                uburnode_somni_pb2.AudioMaterialInput(
                    audio_name="雨声",
                    language="zh",
                    audio_url="https://cdn.example/a.mp3",
                ),
                uburnode_somni_pb2.AudioMaterialInput(),
            ]
        ),
        _context(),
    )
    assert [item.ok for item in created.items] == [True, False]
    assert created.items[1].message == "audio_name 不能为空"
    assert service.create_audios.await_args.args[0] == [
        {"audio_name": "雨声", "language": "zh", "audio_url": "https://cdn.example/a.mp3"},
        {},
    ]

    updated = await rpc.UpdateAudio(
        uburnode_somni_pb2.UpdateAudioReq(
            items=[
                uburnode_somni_pb2.UpdateAudioItem(
                    id="m1",
                    fields=uburnode_somni_pb2.AudioMaterialInput(
                        language="zh",
                        description="只改描述",
                    ),
                )
            ]
        ),
        _context(),
    )
    assert updated.items[0].ok is True
    assert service.update_audios.await_args.args[0] == [
        {"id": "m1", "fields": {"language": "zh", "description": "只改描述"}}
    ]


@pytest.mark.asyncio
async def test_rpc_rejects_batch_over_limit() -> None:
    service = MagicMock()
    service.delete_audios = AsyncMock()
    rpc = AudioRpc(service)
    ctx = _context()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError())
    request = uburnode_somni_pb2.DeleteAudioReq(
        items=[
            uburnode_somni_pb2.DeleteAudioItem(id=str(index))
            for index in range(AUDIO_WRITE_BATCH_LIMIT + 1)
        ]
    )

    with pytest.raises(grpc.aio.AbortError):
        await rpc.DeleteAudio(request, ctx)

    service.delete_audios.assert_not_called()


@pytest.mark.asyncio
async def test_create_rolls_back_mongo_when_locale_save_fails(monkeypatch) -> None:
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    service.collection.delete_one = AsyncMock()
    service.locales.find_one_and_update = AsyncMock(side_effect=RuntimeError("mongo locale down"))
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audios(
        service,
        [
            {
                "audio_name": "测试雨声",
                "language": "zh",
                "audio_url": "https://cdn.example/test.mp3",
            }
        ],
    )

    assert results[0]["ok"] is False
    assert "mongo locale down" in results[0]["message"]
    service.collection.delete_one.assert_awaited()
    assert service.collection.delete_one.await_args.args[0]["_id"] == inserted


@pytest.mark.asyncio
async def test_create_rolls_back_mongo_when_es_index_fails(monkeypatch) -> None:
    """ES 写入失败时回滚测试数据，不碰真实库。"""
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    service.collection.delete_one = AsyncMock()
    service._es_search._client.index = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audios(
        service,
        [
            {
                "audio_name": "测试雨声",
                "language": "zh",
                "audio_url": "https://cdn.example/test.mp3",
            }
        ],
    )

    assert results[0]["ok"] is False
    assert "es down" in results[0]["message"]
    service.collection.delete_one.assert_awaited()
    assert service.collection.delete_one.await_args.args[0]["_id"] == inserted
    service.locales.delete_many.assert_awaited_with({"material_id": str(inserted)})


@pytest.mark.asyncio
async def test_update_restores_mongo_snapshot_when_es_fails(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    before = {"_id": material_id, "status": True, "vip": 0}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "status": False, "vip": 0}
    )
    service.collection.replace_one = AsyncMock()
    service._es_search._client.update = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [{"id": str(material_id), "fields": {"status": False}}],
    )

    assert results[0]["ok"] is False
    assert "es down" in results[0]["message"]
    restored = service.collection.replace_one.await_args.args[1]
    assert restored["status"] is True
    assert "_id" not in restored


@pytest.mark.asyncio
async def test_delete_restores_mongo_when_es_delete_fails(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    before = {"_id": material_id, "status": True}
    locale = _locale_row(material_id=str(material_id), audio_name="测试删除")
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    service.collection.insert_one = AsyncMock()
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[dict(locale)]))
    )
    service.locales.delete_many = AsyncMock()
    service.locales.insert_many = AsyncMock()
    service._es_search._client.delete = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await delete_audios(service, [{"id": str(material_id)}])

    assert results[0]["ok"] is False
    assert "es down" in results[0]["message"]
    service.collection.insert_one.assert_awaited()
    assert service.collection.insert_one.await_args.args[0]["_id"] == material_id
    service.locales.insert_many.assert_awaited()
