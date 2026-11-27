"""量产标签字典批量增加、更新、删除。"""

from __future__ import annotations

import asyncio
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
        refresh_audio_index=AsyncMock(),
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
    collection.update_one = AsyncMock()
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
async def test_successful_tag_write_refreshes_es_before_clearing_cache(monkeypatch) -> None:
    service = _writer()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=ObjectId()))
    order: list[str] = []

    async def _refresh() -> None:
        order.append("refresh")

    async def _clear(_redis: object) -> int:
        order.append("clear")
        return 1

    service._es_search.refresh_audio_index = AsyncMock(side_effect=_refresh)
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        _clear,
    )

    results = await create_audio_tags(
        service,
        [
            {
                "type": "content_form",
                "code": "steady_rain",
                "language": "zh",
                "name": "中雨",
            }
        ],
    )
    assert results[0]["ok"] is True
    assert order == ["refresh", "clear"]


@pytest.mark.asyncio
async def test_tag_write_skips_cache_clear_when_es_refresh_fails(monkeypatch) -> None:
    service = _writer()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=ObjectId()))
    clear_cache = AsyncMock()
    service._es_search.refresh_audio_index = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        clear_cache,
    )

    results = await create_audio_tags(
        service,
        [
            {
                "type": "content_form",
                "code": "steady_rain",
                "language": "zh",
                "name": "中雨",
            }
        ],
    )
    assert results[0]["ok"] is True
    clear_cache.assert_not_awaited()
    service._es_search.clear_content_tag_vectors_cache.assert_not_called()
    assert service._content_form_tag_bank_cache == ["stale"]


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
    kept = service._es_search._client.update.await_args.kwargs["script"]["params"]["fields"]
    assert kept["status"] == "停用"
    assert "name_vector" not in kept
    assert "name_en_vector" not in kept


@pytest.mark.asyncio
async def test_update_rejects_code_change_and_ignores_same_code(monkeypatch) -> None:
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
    service.collection.find_one_and_update = AsyncMock(
        return_value={**before, "name": "细雨"}
    )
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    changed = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"code": "fine_rain"}}],
    )
    assert changed[0]["ok"] is False
    assert changed[0]["message"] == "code 不能修改"
    service.collection.find_one_and_update.assert_not_called()

    same = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"code": "steady_rain", "name": "细雨"}}],
    )
    assert same[0]["ok"] is True
    stored = service.collection.find_one_and_update.await_args.args[1]["$set"]
    assert "code" not in stored
    assert stored["name"] == "细雨"


class _AsyncRows:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __aiter__(self):
        async def generate():
            for row in self._rows:
                yield row

        return generate()


@pytest.mark.asyncio
async def test_renaming_content_form_tag_refreshes_locale_text(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    material_id = ObjectId()
    locale_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "steady_rain",
        "language": "zh",
        "name": "雨声",
        "status": "启用",
    }
    renamed = {**before, "name": "细雨"}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value=dict(renamed))
    service.collection.find = MagicMock(
        return_value=_AsyncRows(
            [
                {
                    "_id": tag_id,
                    "type": "content_form",
                    "status": "启用",
                    "code": "steady_rain",
                    "language": "zh",
                    "name": "细雨",
                }
            ]
        )
    )
    materials = MagicMock()
    materials.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(return_value=[{"_id": material_id, "status": True}])
        )
    )
    materials.find_one = AsyncMock(
        return_value={
            "_id": material_id,
            "status": False,
            "content_form_tags": [{"tag_id": str(tag_id)}],
        }
    )
    locales = MagicMock()
    locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {
                        "_id": locale_id,
                        "material_id": str(material_id),
                        "language": "zh",
                        "audio_name": "夜雨",
                    }
                ]
            )
        )
    )
    service._materials = lambda: materials
    service._locales = lambda: locales
    service._es_search.material_locales_index = "somni_audio_material_locales"
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"name": "细雨"}}],
    )

    assert results[0]["ok"] is True
    locale_patch = next(
        call.kwargs
        for call in service._es_search._client.update.await_args_list
        if call.kwargs["index"] == "somni_audio_material_locales"
    )
    assert locale_patch["script"]["params"]["fields"]["description_text"] == "夜雨 细雨"
    assert locale_patch["script"]["params"]["fields"]["material_status"] is False
    materials.find_one.assert_awaited()


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
async def test_create_tag_reports_id_when_post_insert_compensation_fails(
    monkeypatch,
) -> None:
    service = _writer()
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    service.collection.delete_one = AsyncMock(side_effect=RuntimeError("mongo delete down"))
    service._es_search._client.index = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audio_tags(
        service,
        [
            {
                "type": "mechanism",
                "code": "noise_masking",
                "language": "zh",
                "name": "噪音遮蔽",
            }
        ],
    )

    assert results[0]["ok"] is False
    assert results[0]["id"] == str(inserted)
    assert "es down" in results[0]["message"]


@pytest.mark.asyncio
async def test_create_tag_rolls_back_when_cancelled_after_insert(monkeypatch) -> None:
    service = _writer()
    inserted = ObjectId()
    service.collection.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id=inserted))
    service.collection.delete_one = AsyncMock()
    service._es_search._client.index = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await create_audio_tags(
            service,
            [
                {
                    "type": "mechanism",
                    "code": "noise_masking",
                    "language": "zh",
                    "name": "噪音遮蔽",
                }
            ],
        )

    service.collection.delete_one.assert_awaited_once_with(
        {"_id": inserted, "version": 0}
    )


@pytest.mark.asyncio
async def test_create_tag_rolls_back_when_insert_await_is_cancelled(monkeypatch) -> None:
    service = _writer()
    inserted_doc: dict = {}

    async def cancel_insert(doc: dict) -> None:
        inserted_doc.update(doc)
        raise asyncio.CancelledError()

    service.collection.insert_one = AsyncMock(side_effect=cancel_insert)
    service.collection.delete_one = AsyncMock()
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await create_audio_tags(
            service,
            [
                {
                    "type": "mechanism",
                    "code": "noise_masking",
                    "language": "zh",
                    "name": "噪音遮蔽",
                }
            ],
        )

    assert isinstance(inserted_doc["_id"], ObjectId)
    service.collection.delete_one.assert_awaited_once_with(
        {"_id": inserted_doc["_id"], "version": 0}
    )


@pytest.mark.asyncio
async def test_create_tag_reports_id_when_insert_error_compensation_fails(
    monkeypatch,
) -> None:
    service = _writer()
    inserted_doc: dict = {}

    async def fail_insert(doc: dict) -> None:
        inserted_doc.update(doc)
        raise RuntimeError("mongo timeout")

    service.collection.insert_one = AsyncMock(side_effect=fail_insert)
    service.collection.delete_one = AsyncMock(side_effect=RuntimeError("mongo delete down"))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await create_audio_tags(
        service,
        [
            {
                "type": "mechanism",
                "code": "noise_masking",
                "language": "zh",
                "name": "噪音遮蔽",
            }
        ],
    )

    assert results[0]["ok"] is False
    assert results[0]["id"] == str(inserted_doc["_id"])
    service.collection.delete_one.assert_awaited_once_with(
        {"_id": inserted_doc["_id"], "version": 0}
    )


@pytest.mark.asyncio
async def test_update_tag_restores_snapshot_when_cancelled(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "mechanism",
        "code": "noise_masking",
        "language": "zh",
        "name": "噪音遮蔽",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(
        return_value={**before, "name": "环境噪音遮蔽"}
    )
    service._es_search._client.update = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await update_audio_tags(
            service,
            [{"id": str(tag_id), "fields": {"name": "环境噪音遮蔽"}}],
        )

    reverts = [
        call.args
        for call in service.collection.update_one.await_args_list
        if call.args[0].get("_id") == tag_id and "_ops.name" in call.args[0]
    ]
    assert len(reverts) == 1
    assert reverts[0][1]["$set"] == {"name": "噪音遮蔽"}


@pytest.mark.asyncio
async def test_update_tag_restores_snapshot_when_mongo_await_is_cancelled(
    monkeypatch,
) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "mechanism",
        "code": "noise_masking",
        "language": "zh",
        "name": "噪音遮蔽",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await update_audio_tags(
            service,
            [{"id": str(tag_id), "fields": {"name": "环境噪音遮蔽"}}],
        )

    reverts = [
        call.args
        for call in service.collection.update_one.await_args_list
        if call.args[0].get("_id") == tag_id and "_ops.name" in call.args[0]
    ]
    assert len(reverts) == 1
    assert reverts[0][1]["$set"] == {"name": "噪音遮蔽"}


@pytest.mark.asyncio
async def test_delete_tag_restores_snapshot_when_cancelled(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "mechanism",
        "code": "noise_masking",
        "language": "zh",
        "name": "噪音遮蔽",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    service.collection.insert_one = AsyncMock()
    service._es_search._client.delete = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await delete_audio_tags(service, [{"id": str(tag_id)}])

    service.collection.insert_one.assert_awaited_once()
    assert service.collection.insert_one.await_args.args[0]["_id"] == tag_id


@pytest.mark.asyncio
async def test_delete_tag_restores_snapshot_when_mongo_await_is_cancelled(
    monkeypatch,
) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "mechanism",
        "code": "noise_masking",
        "language": "zh",
        "name": "噪音遮蔽",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.delete_one = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await delete_audio_tags(service, [{"id": str(tag_id)}])

    service.collection.insert_one.assert_awaited_once()
    assert service.collection.insert_one.await_args.args[0]["_id"] == tag_id


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


@pytest.mark.asyncio
async def test_tag_update_refreshes_locales_again_after_rollback(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "steady_rain",
        "language": "zh",
        "name": "雨声",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(
        return_value={**before, "name": "细雨"}
    )
    events: list[str] = []

    async def _refresh(
        _service: object,
        _codes: set[str],
        _tag_ids: set[str],
    ) -> None:
        events.append("refresh")
        if events.count("refresh") == 1:
            raise RuntimeError("locale es down")

    async def _restore(*_args: object, **_kwargs: object) -> None:
        events.append("restore")

    service.collection.replace_one = AsyncMock(side_effect=_restore)
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.refresh_locales_for_content_codes",
        _refresh,
    )
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service, [{"id": str(tag_id), "fields": {"name": "细雨"}}]
    )

    assert results[0]["ok"] is False
    assert events == ["refresh", "refresh"]


@pytest.mark.asyncio
async def test_unchanged_name_still_refreshes_when_locale_refresh_is_pending(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "rain",
        "language": "zh",
        "name": "夜雨",
        "status": "启用",
        "locale_refresh_pending": True,
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value=dict(before))
    refreshed = AsyncMock()
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.refresh_locales_for_content_codes",
        refreshed,
    )
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service, [{"id": str(tag_id), "fields": {"name": "夜雨"}}]
    )

    assert results[0]["ok"] is True
    refreshed.assert_awaited_once()
    assert refreshed.await_args.args[1] == {"rain"}
    cleared = service.collection.update_one.await_args.args[1]
    assert cleared == {"$set": {"locale_refresh_pending": False}}
