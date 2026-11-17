"""量产标签字典批量增加、更新、删除。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.server.somni.audio.rpc import AudioRpc
from app.server.somni.audio.tag_writes import (
    audio_tag_input_to_fields,
    create_audio_tags,
    delete_audio_tags,
    update_audio_tags,
)
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def _writer() -> SimpleNamespace:
    es = SimpleNamespace(
        tag_dictionary_index="somni_audio_tag_dictionary",
        _client=SimpleNamespace(
            index=AsyncMock(),
            update=AsyncMock(),
            delete=AsyncMock(),
            get=AsyncMock(
                return_value=SimpleNamespace(
                    body={"_source": {"name_vector": [0.3, 0.4], "name_en_vector": [0.5]}}
                )
            ),
        ),
        clear_content_tag_vectors_cache=MagicMock(),
    )
    encoder = SimpleNamespace(encode_one=AsyncMock(return_value=[0.2, 0.4]))
    collection = MagicMock()
    collection.find_one = AsyncMock(
        return_value={
            "_id": ObjectId(),
            "type": "content_form",
            "code": "steady_rain",
            "language": "zh",
            "name": "中雨",
            "status": "启用",
        }
    )
    collection.replace_one = AsyncMock()
    collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=ObjectId()))
    service = SimpleNamespace(
        collection=collection,
        _es_search=es,
        _encoder=encoder,
        _catalog_cache=SimpleNamespace(_redis=object()),
        _content_form_tag_bank_cache=["stale"],
        _content_form_tag_labels_cache={"stale": {}},
        _tags=lambda: None,
    )
    service._tags = lambda: service.collection
    return service


@pytest.mark.asyncio
async def test_create_tag_writes_language_vector(monkeypatch) -> None:
    service = _writer()
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audio_tags(
        service,
        [
            {},
            {
                "type": "content_form",
                "code": "steady_rain",
                "language": "ZH",
                "name": "中雨",
            },
        ],
    )

    assert results[0]["ok"] is False
    assert "不能为空" in results[0]["message"]
    assert results[1]["ok"] is True
    doc = service.collection.insert_one.await_args.args[0]
    assert doc["language"] == "zh"
    assert doc["status"] == "启用"
    assert doc["parent_id"] is None
    indexed = service._es_search._client.index.await_args.kwargs
    assert indexed["index"] == "somni_audio_tag_dictionary"
    assert indexed["document"]["name_vector"] == [0.2, 0.4]
    assert "name_en_vector" not in indexed["document"]
    assert service._encoder.encode_one.await_args.kwargs["language"] == "zh"
    assert service._content_form_tag_bank_cache is None
    service._es_search.clear_content_tag_vectors_cache.assert_called_once()


@pytest.mark.asyncio
async def test_create_english_tag_uses_name_en_vector(monkeypatch) -> None:
    service = _writer()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=ObjectId()))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audio_tags(
        service,
        [{"type": "content_form", "code": "steady_rain", "language": "en", "name": "Steady Rain"}],
    )

    assert results[0]["ok"] is True
    indexed = service._es_search._client.index.await_args.kwargs["document"]
    assert indexed["name_en_vector"] == [0.2, 0.4]
    assert "name_vector" not in indexed
    assert service._encoder.encode_one.await_args.kwargs["language"] == "en"


@pytest.mark.asyncio
async def test_update_name_recomputes_and_status_keeps_vector(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "steady_rain",
        "language": "zh",
        "name": "稳定雨声",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value=dict(before))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    renamed = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"name": "稳定雨声"}}],
    )
    assert renamed[0]["ok"] is True
    service._es_search._client.index.assert_not_called()
    assert service._encoder.encode_one.await_args.args[0] == "稳定雨声"
    renamed_doc = service._es_search._client.update.await_args.kwargs["script"]["params"]
    assert renamed_doc["fields"]["name"] == "稳定雨声"
    assert renamed_doc["fields"]["name_vector"] == [0.2, 0.4]
    assert "name_en_vector" in renamed_doc["remove"]

    service._encoder.encode_one.reset_mock()
    service._es_search._client.update.reset_mock()
    paused_doc = {
        "_id": tag_id,
        "type": "content_form",
        "code": "steady_rain",
        "language": "zh",
        "name": "稳定雨声",
        "status": "停用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(paused_doc))
    service.collection.find_one_and_update = AsyncMock(return_value=dict(paused_doc))
    paused = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"status": "停用"}}],
    )
    assert paused[0]["ok"] is True
    service._encoder.encode_one.assert_not_called()
    service._es_search._client.index.assert_not_called()
    kept = service._es_search._client.update.await_args.kwargs["doc"]
    assert kept["status"] == "停用"
    assert "name_vector" not in kept
    assert "name_en_vector" not in kept


@pytest.mark.asyncio
async def test_duplicate_tag_is_item_error(monkeypatch) -> None:
    service = _writer()
    service.collection.insert_one = AsyncMock(side_effect=DuplicateKeyError("dup"))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audio_tags(
        service,
        [{"type": "content_form", "code": "rain", "language": "zh", "name": "雨"}],
    )

    assert results[0]["ok"] is False
    assert results[0]["message"] == "标签已存在：type + code + language"


def test_tag_input_keeps_only_present_fields() -> None:
    message = uburnode_somni_pb2.AudioTagInput(name="雨声", parent_id="")
    assert audio_tag_input_to_fields(message) == {"name": "雨声", "parent_id": None}


@pytest.mark.asyncio
async def test_rpc_create_audio_tag_passes_partial_fields() -> None:
    service = MagicMock()
    service.create_audio_tags = AsyncMock(
        return_value=[{"index": 0, "ok": True, "id": "t1", "message": ""}]
    )
    rpc = AudioRpc(service)

    response = await rpc.CreateAudioTag(
        uburnode_somni_pb2.CreateAudioTagReq(
            items=[
                uburnode_somni_pb2.AudioTagInput(
                    type="content_form",
                    code="rain",
                    language="zh",
                    name="雨声",
                )
            ]
        ),
        MagicMock(),
    )

    assert response.items[0].id == "t1"
    assert service.create_audio_tags.await_args.args[0] == [
        {"type": "content_form", "code": "rain", "language": "zh", "name": "雨声"}
    ]


@pytest.mark.asyncio
async def test_create_tag_rolls_back_mongo_when_es_fails(monkeypatch) -> None:
    service = _writer()
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    service.collection.delete_one = AsyncMock()
    service._es_search._client.index = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audio_tags(
        service,
        [
            {
                "type": "content_form",
                "code": "test_rain",
                "language": "zh",
                "name": "测试雨",
            }
        ],
    )

    assert results[0]["ok"] is False
    assert "es down" in results[0]["message"]
    service.collection.delete_one.assert_awaited()
    assert service.collection.delete_one.await_args.args[0]["_id"] == inserted


@pytest.mark.asyncio
async def test_delete_tag_restores_mongo_when_es_fails(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "test_rain",
        "language": "zh",
        "name": "测试雨",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    service.collection.insert_one = AsyncMock()
    service._es_search._client.delete = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await delete_audio_tags(service, [{"id": str(tag_id)}])

    assert results[0]["ok"] is False
    assert "es down" in results[0]["message"]
    service.collection.insert_one.assert_awaited()
    assert service.collection.insert_one.await_args.args[0]["_id"] == tag_id
