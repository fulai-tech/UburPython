"""量产标签字典批量增加、更新、删除。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

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
        material_locales_index="somni_audio_material_locales",
        audio_index="somni_audio_materials",
        refresh_audio_index=AsyncMock(),
        _client=SimpleNamespace(
            index=AsyncMock(),
            update=AsyncMock(),
            delete=AsyncMock(),
            bulk=AsyncMock(return_value={"errors": False, "items": []}),
            get=AsyncMock(
                return_value=SimpleNamespace(
                    body={"_source": {"name_vector": [0.3, 0.4], "name_en_vector": [0.5]}}
                )
            ),
        ),
        clear_content_tag_vectors_cache=MagicMock(),
    )
    encoder = SimpleNamespace(
        encode_one=AsyncMock(return_value=[0.2, 0.4]),
        encode=AsyncMock(
            side_effect=lambda texts, language="zh": [[0.2, 0.4] for _ in texts]
        ),
    )
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
    materials = MagicMock()
    materials.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[]))
    )
    materials.find_one = AsyncMock(return_value=None)
    locales = MagicMock()
    locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[]))
    )
    service = SimpleNamespace(
        collection=collection,
        materials=materials,
        locales=locales,
        _es_search=es,
        _encoder=encoder,
        _catalog_cache=SimpleNamespace(_redis=object()),
        _content_form_tag_bank_cache=["stale"],
        _content_form_tag_labels_cache={"stale": {}},
        _tags=lambda: None,
        _materials=lambda: None,
        _locales=lambda: None,
    )
    service._tags = lambda: service.collection
    service._materials = lambda: service.materials
    service._locales = lambda: service.locales
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


@pytest.mark.asyncio
async def test_failed_code_change_refreshes_old_and_new_code_after_rollback(
    monkeypatch,
) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "old_rain",
        "language": "zh",
        "name": "雨",
        "status": "启用",
    }
    saved = {**before, "code": "new_rain"}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value=dict(saved))
    service._es_search._client.update = AsyncMock(side_effect=RuntimeError("es down"))
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"code": "new_rain"}}],
    )

    assert results[0]["ok"] is False
    queried_codes = [
        call.args[0]["content_form_tags.code"] for call in service.materials.find.call_args_list
    ]
    assert queried_codes == ["old_rain", "new_rain"]


@pytest.mark.asyncio
async def test_update_content_form_name_refreshes_linked_material_locales(
    monkeypatch,
) -> None:
    service = _writer()
    tag_id = ObjectId()
    material_id = ObjectId()
    locale_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "steady_rain",
        "language": "zh",
        "name": "旧雨名",
        "status": "启用",
    }
    saved = {**before, "name": "新雨名"}
    material = {
        "_id": material_id,
        "content_form_tags": [{"code": "steady_rain"}],
        "status": True,
    }
    locale = {
        "_id": locale_id,
        "material_id": str(material_id),
        "language": "zh",
        "audio_name": "雨曲",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value=dict(saved))
    service.materials.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[material]))
    )
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[locale]))
    )
    service.collection.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {
                        "type": "content_form",
                        "code": "steady_rain",
                        "language": "zh",
                        "name": "新雨名",
                    }
                ]
            )
        )
    )
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"name": "新雨名"}}],
    )

    assert results[0]["ok"] is True
    assert call({"content_form_tags.code": "steady_rain"}) in service.materials.find.call_args_list
    # 纯改名只刷一次，且一次 bulk 写完；写后复查一致，不再重写
    service._es_search._client.bulk.assert_awaited_once()
    operations = service._es_search._client.bulk.await_args.kwargs["operations"]
    assert operations[0] == {
        "update": {
            "_index": "somni_audio_material_locales",
            "_id": str(locale_id),
            "retry_on_conflict": 3,
        }
    }
    # 只改派生检索字段，不带标题，避免覆盖并发编辑；缺文档时才 upsert 整份
    params = operations[1]["script"]["params"]
    assert params["fields"] == {"description_text": "新雨名", "description_vector": [0.2, 0.4]}
    assert params["remove"] == ["description_vector_en"]
    assert operations[1]["upsert"]["audio_name"] == "雨曲"


@pytest.mark.asyncio
async def test_rename_batches_dictionary_encode_and_bulk_across_materials(
    monkeypatch,
) -> None:
    """多条素材共享同一检索文本：词典查一次、文本只编码一次、ES 一次 bulk。"""
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "rain",
        "language": "zh",
        "name": "旧雨",
        "status": "启用",
    }
    materials = [
        {"_id": ObjectId(), "content_form_tags": [{"code": "rain"}]} for _ in range(5)
    ]
    locales = [
        {"_id": ObjectId(), "material_id": str(m["_id"]), "language": lang, "audio_name": "x"}
        for m in materials
        for lang in ("zh", "en")
    ]
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value={**before, "name": "新雨"})
    service.materials.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=materials))
    )
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=locales))
    )
    service.collection.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"type": "content_form", "code": "rain", "language": "zh", "name": "新雨"}
                ]
            )
        )
    )
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"name": "新雨"}}],
    )

    assert results[0]["ok"] is True
    # 只刷 zh：中文词典改名不影响英文行；写入一轮 + 写后复查一轮，各查一次语言行与词典
    assert service.locales.find.call_count == 2
    assert service.collection.find.call_count == 2
    service._encoder.encode.assert_awaited_once()
    assert service._encoder.encode.await_args.args[0] == ["新雨"]
    service._es_search._client.bulk.assert_awaited_once()
    operations = service._es_search._client.bulk.await_args.kwargs["operations"]
    assert len(operations) == 10
    assert all(doc["upsert"]["language"] == "zh" for doc in operations[1::2])


@pytest.mark.asyncio
async def test_concurrent_rename_during_encode_recomputes_with_latest_name(
    monkeypatch,
) -> None:
    """编码期间词典又被改名：按最新名称重算，不把旧文本写回。"""
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "rain",
        "language": "zh",
        "name": "旧雨",
        "status": "启用",
    }
    material = {"_id": ObjectId(), "content_form_tags": [{"code": "rain"}]}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value={**before, "name": "新雨"})
    service.materials.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[material]))
    )
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"_id": ObjectId(), "material_id": str(material["_id"]), "language": "zh"}
                ]
            )
        )
    )
    snapshots = iter(["新雨", "更新的雨", "更新的雨"])

    def _dictionary(*_args, **_kwargs):
        name = next(snapshots)
        return SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"type": "content_form", "code": "rain", "language": "zh", "name": name}
                ]
            )
        )

    service.collection.find = MagicMock(side_effect=_dictionary)
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"name": "新雨"}}],
    )

    assert results[0]["ok"] is True
    encoded = [call.args[0] for call in service._encoder.encode.await_args_list]
    assert encoded == [["新雨"], ["更新的雨"]]
    operations = service._es_search._client.bulk.await_args.kwargs["operations"]
    assert operations[1]["script"]["params"]["fields"]["description_text"] == "更新的雨"


def _rename_voice_tag(service: SimpleNamespace) -> ObjectId:
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "voice",
        "language": "zh",
        "name": "旧人声",
        "status": "启用",
    }
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(
        return_value={**before, "name": "人声引导"}
    )
    service.collection.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"type": "content_form", "code": "voice", "language": "zh", "name": "人声引导"},
                    {"type": "content_form", "code": "rain", "language": "zh", "name": "雨声"},
                ]
            )
        )
    )
    return tag_id


@pytest.mark.asyncio
async def test_material_tag_change_during_refresh_is_rewritten_with_latest_tags(
    monkeypatch,
) -> None:
    """刷新编码期间另一请求把素材标签从人声引导改成雨声：写后复查按雨声重写，不回退。"""
    service = _writer()
    tag_id = _rename_voice_tag(service)
    material_id = ObjectId()
    locale_id = ObjectId()
    current = {"_id": material_id, "content_form_tags": [{"code": "voice"}]}
    service.materials.find = MagicMock(
        side_effect=lambda *_a, **_k: SimpleNamespace(
            to_list=AsyncMock(return_value=[dict(current)])
        )
    )
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[{"_id": locale_id, "material_id": str(material_id), "language": "zh"}]
            )
        )
    )

    async def _encode(texts, language="zh"):
        current["content_form_tags"] = [{"code": "rain"}]
        return [[0.2, 0.4] for _ in texts]

    service._encoder.encode = AsyncMock(side_effect=_encode)
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service, [{"id": str(tag_id), "fields": {"name": "人声引导"}}]
    )

    assert results[0]["ok"] is True
    written = [
        op["script"]["params"]["fields"]["description_text"]
        for bulk_call in service._es_search._client.bulk.await_args_list
        for op in bulk_call.kwargs["operations"][1::2]
    ]
    assert written == ["人声引导", "雨声"]


@pytest.mark.asyncio
async def test_locale_deleted_during_refresh_removes_upserted_document(monkeypatch) -> None:
    """刷新编码期间素材被删：upsert 可能把文档写回，写后复查发现语言行已删就删掉。"""
    service = _writer()
    tag_id = _rename_voice_tag(service)
    material_id = ObjectId()
    locale_id = ObjectId()
    rows = [{"_id": locale_id, "material_id": str(material_id), "language": "zh"}]
    service.materials.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[{"_id": material_id, "content_form_tags": [{"code": "voice"}]}]
            )
        )
    )
    service.locales.find = MagicMock(
        side_effect=lambda *_a, **_k: SimpleNamespace(to_list=AsyncMock(return_value=list(rows)))
    )

    async def _encode(texts, language="zh"):
        rows.clear()
        return [[0.2, 0.4] for _ in texts]

    service._encoder.encode = AsyncMock(side_effect=_encode)
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service, [{"id": str(tag_id), "fields": {"name": "人声引导"}}]
    )

    assert results[0]["ok"] is True
    service._es_search._client.bulk.assert_awaited_once()
    assert (
        call(index="somni_audio_material_locales", id=str(locale_id))
        in service._es_search._client.delete.await_args_list
    )


@pytest.mark.asyncio
async def test_bulk_item_error_fails_tag_update(monkeypatch) -> None:
    service = _writer()
    tag_id = ObjectId()
    before = {
        "_id": tag_id,
        "type": "content_form",
        "code": "rain",
        "language": "zh",
        "name": "旧雨",
        "status": "启用",
    }
    material = {"_id": ObjectId(), "content_form_tags": [{"code": "rain"}]}
    service.collection.find_one = AsyncMock(return_value=dict(before))
    service.collection.find_one_and_update = AsyncMock(return_value={**before, "name": "新雨"})
    service.materials.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[material]))
    )
    service.locales.find = MagicMock(
        return_value=SimpleNamespace(
            to_list=AsyncMock(
                return_value=[
                    {"_id": ObjectId(), "material_id": str(material["_id"]), "language": "zh"}
                ]
            )
        )
    )
    service.collection.find = MagicMock(
        return_value=SimpleNamespace(to_list=AsyncMock(return_value=[]))
    )
    service._es_search._client.bulk = AsyncMock(
        return_value={"errors": True, "items": [{"index": {"error": {"type": "mapper"}}}]}
    )
    monkeypatch.setattr(
        "app.server.somni.audio.tag_writes.clear_somni_audio_catalog_cache",
        AsyncMock(),
    )

    results = await update_audio_tags(
        service,
        [{"id": str(tag_id), "fields": {"name": "新雨"}}],
    )

    assert results[0]["ok"] is False
    assert "语言索引批量写入失败" in results[0]["message"]
    service.collection.replace_one.assert_awaited()
