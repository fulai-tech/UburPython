"""量产音频原料批量增加、更新、删除。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from bson import ObjectId
from elasticsearch import NotFoundError

from app.server.somni.audio.rpc import AudioRpc
from app.server.somni.audio.writes import (
    AUDIO_WRITE_BATCH_LIMIT,
    _reindex_material_locales,
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
            get=AsyncMock(
                return_value=SimpleNamespace(body={"_source": {"description_vector": [0.9, 0.1]}})
            ),
        ),
    )
    encoder = SimpleNamespace(encode_one=AsyncMock(return_value=[0.2, 0.4]))
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value={"_id": ObjectId(), "status": True})
    collection.replace_one = AsyncMock()
    collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=ObjectId()))
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
    locales.find = MagicMock(return_value=SimpleNamespace(to_list=AsyncMock(return_value=[])))
    return SimpleNamespace(
        collection=collection,
        locales=locales,
        locale_row=row,
        _es_search=es,
        _encoder=encoder,
        _catalog_cache=SimpleNamespace(_redis=object()),
        _retrieval=SimpleNamespace(clear_sleep_stage_cache=AsyncMock()),
        _materials=lambda: None,
        _locales=lambda: None,
    )


def _bind(service: SimpleNamespace) -> SimpleNamespace:
    service._materials = lambda: service.collection
    service._locales = lambda: service.locales
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
                "origin_file_name": "rain.mp3",
                "acoustic_metrics": {"voice_presence_ratio": 0.4},
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
    assert material_indexed["document"]["origin_file_name"] == "rain.mp3"
    assert material_indexed["document"]["acoustic_metrics"] == {"voice_presence_ratio": 0.4}
    locale_indexed = [
        call.kwargs for call in service._es_search._client.index.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    ][0]
    assert locale_indexed["document"]["audio_name"] == "雨声"
    assert locale_indexed["document"]["material_status"] is False
    assert locale_indexed["document"]["description_vector"] == [0.2, 0.4]
    assert service._encoder.encode_one.await_args.kwargs["language"] == "zh"
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
    # 只改语言表字段时重算语言向量，不改原料索引展示字段
    assert service._encoder.encode_one.await_args.args[0]
    patched = service._es_search._client.update.await_args.kwargs
    assert patched["index"] == "somni_audio_material_locales"
    assert patched["script"]["params"]["fields"]["summary"] == "短"


@pytest.mark.asyncio
async def test_update_acoustic_metrics_preserves_unspecified_measurements(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    before = {
        "_id": material_id,
        "status": True,
        "acoustic_metrics": {
            "duration_seconds": 90.0,
            "tempo_bpm": 72.0,
            "voice_presence_ratio": 0.2,
        },
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(
        return_value={
            **before,
            "acoustic_metrics": {
                **before["acoustic_metrics"],
                "tempo_bpm": 60.0,
            },
        }
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [{"id": str(material_id), "fields": {"acoustic_metrics": {"tempo_bpm": 60.0}}}],
    )

    assert results[0]["ok"] is True
    fields = service.collection.find_one_and_update.await_args.args[1]["$set"]
    assert fields["acoustic_metrics.tempo_bpm"] == 60.0
    assert "acoustic_metrics" not in fields


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
async def test_update_recomputes_vector_when_title_description_or_tags_are_sent(
    monkeypatch,
) -> None:
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
    encoded_texts = [call.args[0] for call in service._encoder.encode_one.await_args_list]
    assert encoded_texts[0] == "同样的描述"
    assert "雨声" in encoded_texts[1:]
    assert all("同样的描述" not in text for text in encoded_texts[1:])
    stored = service.collection.find_one_and_update.await_args.args[1]["$set"]
    assert stored["embedding"] == [0.2, 0.4]
    locale_patch = next(
        call.kwargs
        for call in service._es_search._client.update.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    )
    fields = locale_patch["script"]["params"]["fields"]
    assert fields["description"] == "同样的描述"
    assert fields["description_text"] == "雨声"
    assert fields["description_vector"] == [0.2, 0.4]
    assert "embedding" not in fields


class _AsyncRows:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __aiter__(self):
        async def generate():
            for row in self._rows:
                yield row

        return generate()


@pytest.mark.asyncio
async def test_content_form_tag_change_reads_dictionary_once(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    zh = _locale_row(material_id=str(material_id), language="zh", audio_name="细雨")
    en = _locale_row(material_id=str(material_id), language="en", audio_name="Rain")
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[zh, en]))
    )
    service.collection.find_one_and_update = AsyncMock(
        return_value={
            "_id": material_id,
            "status": True,
            "content_form_tags": [{"code": "rain"}, {"code": "voice"}],
        }
    )
    find = MagicMock(
        return_value=_AsyncRows(
            [
                {
                    "type": "content_form",
                    "status": "启用",
                    "code": "rain",
                    "language": "zh",
                    "name": "雨声",
                },
                {
                    "type": "content_form",
                    "status": "启用",
                    "code": "rain",
                    "language": "en",
                    "name": "Rain Sound",
                },
                {
                    "type": "content_form",
                    "status": "启用",
                    "code": "voice",
                    "language": "zh",
                    "name": "人声",
                },
            ]
        )
    )
    service._tags = lambda: SimpleNamespace(find=find)
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
                    "content_form_tags": [{"code": "rain"}, {"code": "voice"}],
                },
            }
        ],
    )

    assert results[0]["ok"] is True
    assert find.call_count == 1
    assert "language" not in find.call_args.args[0]
    texts = [
        call.kwargs["script"]["params"]["fields"]["description_text"]
        for call in service._es_search._client.update.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    ]
    assert texts == ["细雨 雨声 人声", "Rain Rain Sound"]


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

    locale_id = ObjectId()
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[{"_id": locale_id}]))
    )

    results = await update_audios(
        service,
        [{"id": str(material_id), "fields": {"status": False}}],
    )

    assert results[0]["ok"] is True
    service._encoder.encode_one.assert_not_called()
    service._es_search._client.index.assert_not_called()
    docs = [call.kwargs["doc"] for call in service._es_search._client.update.await_args_list]
    assert {"status": False} in docs
    assert {"material_status": False} in docs
    assert all("description_vector" not in doc for doc in docs)
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
    service._es_search._client.delete.assert_awaited_once()
    clear_cache.assert_awaited_once()


def test_content_form_parent_is_stored_as_parent_id() -> None:
    message = uburnode_somni_pb2.AudioMaterialInput()
    tag = message.content_form_tags.items.add()
    tag.code = "rain"
    tag.parent_tag_id = "parent-1"

    assert audio_material_input_to_fields(message) == {
        "content_form_tags": [{"code": "rain", "parent_id": "parent-1"}]
    }


def test_acoustic_metrics_object_round_trips_null_tempo() -> None:
    message = uburnode_somni_pb2.AudioMaterialInput()
    message.acoustic_metrics.duration_seconds = 90
    message.acoustic_metrics.voice_presence_ratio = 0.2
    message.acoustic_metrics.tempo_no_beat = True

    assert audio_material_input_to_fields(message) == {
        "acoustic_metrics": {
            "duration_seconds": 90.0,
            "voice_presence_ratio": 0.2,
            "tempo_bpm": None,
        }
    }


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
async def test_create_reports_inserted_id_when_compensation_fails(monkeypatch) -> None:
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    service.collection.delete_one = AsyncMock(side_effect=RuntimeError("mongo delete down"))
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
    assert results[0]["id"] == str(inserted)
    assert "es down" in results[0]["message"]


@pytest.mark.asyncio
async def test_create_rolls_back_when_the_call_is_cancelled(monkeypatch) -> None:
    service = _bind(_writer())
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    service.collection.delete_one = AsyncMock()
    service.locales.find_one_and_update = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await create_audios(
            service,
            [
                {
                    "audio_name": "测试雨声",
                    "language": "zh",
                    "audio_url": "https://cdn.example/test.mp3",
                }
            ],
        )

    service.collection.delete_one.assert_awaited()
    assert service.collection.delete_one.await_args.args[0]["_id"] == inserted


@pytest.mark.asyncio
async def test_create_rolls_back_when_insert_await_is_cancelled(monkeypatch) -> None:
    service = _bind(_writer())
    inserted_doc: dict = {}

    async def cancel_insert(doc: dict) -> None:
        inserted_doc.update(doc)
        raise asyncio.CancelledError()

    service.collection.insert_one = AsyncMock(side_effect=cancel_insert)
    service.collection.delete_one = AsyncMock()
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await create_audios(
            service,
            [
                {
                    "audio_name": "测试雨声",
                    "language": "zh",
                    "audio_url": "https://cdn.example/test.mp3",
                }
            ],
        )

    assert isinstance(inserted_doc["_id"], ObjectId)
    service.collection.delete_one.assert_awaited_once_with({"_id": inserted_doc["_id"]})


@pytest.mark.asyncio
async def test_create_reports_id_when_insert_error_compensation_fails(monkeypatch) -> None:
    service = _bind(_writer())
    inserted_doc: dict = {}

    async def fail_insert(doc: dict) -> None:
        inserted_doc.update(doc)
        raise RuntimeError("mongo timeout")

    service.collection.insert_one = AsyncMock(side_effect=fail_insert)
    service.collection.delete_one = AsyncMock(side_effect=RuntimeError("mongo delete down"))
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
    assert results[0]["id"] == str(inserted_doc["_id"])
    service.collection.delete_one.assert_awaited_once_with({"_id": inserted_doc["_id"]})


@pytest.mark.asyncio
async def test_update_restores_snapshot_when_mongo_await_is_cancelled(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    before = {"_id": material_id, "status": True, "vip": 0}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await update_audios(
            service,
            [{"id": str(material_id), "fields": {"status": False}}],
        )

    service.collection.replace_one.assert_awaited_once()
    assert service.collection.replace_one.await_args.args[0]["_id"] == material_id


@pytest.mark.asyncio
async def test_create_compensation_removes_locale_es_document(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    locale_id = ObjectId()
    locale = _locale_row(
        _id=locale_id,
        material_id=str(material_id),
        audio_name="测试雨声",
    )
    service.collection.insert_one = AsyncMock(
        return_value=SimpleNamespace(inserted_id=material_id)
    )
    service.locales.find_one_and_update = AsyncMock(return_value=locale)
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[locale]))
    )
    service._es_search._client.index = AsyncMock(
        side_effect=[None, RuntimeError("locale es down")]
    )
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
    assert any(
        call.kwargs.get("index") == service._es_search.material_locales_index
        and call.kwargs.get("id") == str(locale_id)
        for call in service._es_search._client.delete.await_args_list
    )


@pytest.mark.asyncio
async def test_reindex_removes_blank_language_from_the_locale_index() -> None:
    service = _bind(_writer())
    blank_id = ObjectId()
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {
                        "_id": blank_id,
                        "material_id": "m1",
                        "language": "",
                        "audio_name": "旧标题",
                    }
                ]
            )
        )
    )

    await _reindex_material_locales(service, "m1", {"status": True, "content_form_tags": []})

    service._es_search._client.delete.assert_awaited()
    assert service._es_search._client.delete.await_args.kwargs["id"] == str(blank_id)


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
async def test_update_failure_reindexes_every_locale_from_snapshot(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    zh_id = ObjectId()
    en_id = ObjectId()
    before = {"_id": material_id, "status": True}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "status": False}
    )
    service.collection.replace_one = AsyncMock()
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {
                        "_id": zh_id,
                        "material_id": str(material_id),
                        "language": "zh",
                        "audio_name": "夜雨",
                    },
                    {
                        "_id": en_id,
                        "material_id": str(material_id),
                        "language": "en",
                        "audio_name": "Night",
                    },
                ]
            )
        )
    )
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
    locale_docs = {
        call.kwargs["id"]: call.kwargs["document"]
        for call in service._es_search._client.index.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    }
    assert locale_docs[str(zh_id)]["material_status"] is True
    assert locale_docs[str(en_id)]["material_status"] is True
    assert locale_docs[str(zh_id)]["description_text"] == "夜雨"


@pytest.mark.asyncio
async def test_update_failure_restores_es_material_even_when_tag_dictionary_is_down(
    monkeypatch,
) -> None:
    """词典不可用导致更新失败时，补偿也读不到词典；原料 ES 文档仍须先恢复成旧快照。"""
    service = _bind(_writer())
    material_id = ObjectId()
    before = {
        "_id": material_id,
        "status": True,
        "content_form_tags": [{"code": "rain"}],
    }
    after = {
        "_id": material_id,
        "status": True,
        "content_form_tags": [{"code": "voice"}],
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value=after)
    service.collection.replace_one = AsyncMock()
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[_locale_row(material_id=str(material_id), language="zh")]
            )
        )
    )

    def _tags_down() -> None:
        raise RuntimeError("tags down")

    service._tags = _tags_down
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audios(
        service,
        [
            {
                "id": str(material_id),
                "fields": {"content_form_tags": [{"code": "voice"}]},
            }
        ],
    )

    assert results[0]["ok"] is False
    assert "标签词典不可用" in results[0]["message"]
    restored = [
        call.kwargs["document"]
        for call in service._es_search._client.index.await_args_list
        if call.kwargs["index"] == "somni_audio_materials"
    ]
    assert restored, "原料 ES 文档没有恢复，仍停在新标签"
    assert restored[-1]["content_form_tags"] == [{"code": "rain"}]


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
    service.collection.replace_one.assert_awaited()
    assert service.collection.replace_one.await_args.args[0]["_id"] == material_id
    assert service.collection.replace_one.await_args.kwargs["upsert"] is True
    service.locales.replace_one.assert_awaited()
    assert service.locales.replace_one.await_args.kwargs["upsert"] is True


@pytest.mark.asyncio
async def test_delete_restores_snapshot_when_mongo_await_is_cancelled(monkeypatch) -> None:
    service = _bind(_writer())
    material_id = ObjectId()
    before = {"_id": material_id, "status": True}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.delete_one = AsyncMock(side_effect=asyncio.CancelledError())
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[]))
    )
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await delete_audios(service, [{"id": str(material_id)}])

    service.collection.replace_one.assert_awaited_once()
    assert service.collection.replace_one.await_args.args[0]["_id"] == material_id
    assert service.collection.replace_one.await_args.kwargs["upsert"] is True


@pytest.mark.asyncio
async def test_same_material_update_waits_for_previous_es_write(monkeypatch) -> None:
    """同一条原料的下一次更新要等上一次 ES 写完。"""
    service = _bind(_writer())
    material_id = ObjectId()
    service.collection.find_one = AsyncMock(return_value={"_id": material_id, "status": True})
    service.collection.find_one_and_update = AsyncMock(
        return_value={"_id": material_id, "status": False}
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_update(**_kwargs: object) -> None:
        started.set()
        await release.wait()

    service._es_search._client.update = AsyncMock(side_effect=_slow_update)
    monkeypatch.setattr(
        "app.server.somni.audio.writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    first = asyncio.create_task(
        update_audios(service, [{"id": str(material_id), "fields": {"status": False}}])
    )
    await started.wait()
    reads_while_busy = service.collection.find_one.await_count
    second = asyncio.create_task(
        update_audios(service, [{"id": str(material_id), "fields": {"status": True}}])
    )
    await asyncio.sleep(0.05)
    assert service.collection.find_one.await_count == reads_while_busy
    release.set()
    await first
    await second
    assert service.collection.find_one.await_count > reads_while_busy
