"""量产音频原料批量写入。

原料表/原料索引只存状态、来源、标签和 embedding。
标题、描述、短描述、audio_url、cover_url 与检索向量写语言表/语言索引。
"""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from elasticsearch import NotFoundError
from google.protobuf.json_format import MessageToDict
from loguru import logger
from pymongo import ReturnDocument

from app.cache.somni_audio_catalog_cache import clear_somni_audio_catalog_cache
from app.core.bson_util import bson_to_jsonable, utc_now
from app.core.exceptions import AppError
from app.es.somni_docs import (
    build_locale_description_text,
    extract_sleep_stage_names,
    locale_source_for_es,
)

AUDIO_WRITE_BATCH_LIMIT = 200

_MATERIAL_FIELDS = (
    "status",
    "operation_type",
    "source",
    "vip",
    "created_by",
    "updated_by",
)
_TAG_FIELDS = (
    "sleep_stage_tags",
    "content_form_tags",
    "audio_engineering_tags",
    "evidence_level_tags",
)
_LOCALE_FIELDS = (
    "language",
    "audio_name",
    "audio_url",
    "cover_url",
    "description",
    "summary",
)
_LOCALE_TEXT_FIELDS = ("audio_name", "description", "summary")
_CREATE_MATERIAL_DEFAULTS: dict[str, Any] = {
    "status": True,
    "operation_type": 0,
    "source": "platform",
    "vip": 0,
    "created_by": "",
    "updated_by": "",
    "sleep_stage_tags": [],
    "content_form_tags": [],
    "audio_engineering_tags": [],
    "evidence_level_tags": [],
}
_CREATE_LOCALE_DEFAULTS: dict[str, Any] = {
    "audio_url": "",
    "cover_url": "",
    "description": "",
    "summary": "",
}
_ES_SKIP = {
    "_id",
    "id",
    "embedding",
    "language",
    "audio_name",
    "audio_name_en",
    "description",
    "description_en",
    "summary",
    "short_description",
    "short_description_en",
    "audio_url",
    "cover_url",
    "description_text",
    "description_text_en",
    "description_vector",
    "description_vector_en",
    "mechanism_tags",
    "medical_risk_tags",
}


def audio_material_input_to_fields(message: Any) -> dict[str, Any]:
    """只取出调用方显式设置的字段。"""
    fields: dict[str, Any] = {}
    for name in (*_MATERIAL_FIELDS, *_LOCALE_FIELDS):
        if message.HasField(name):
            value = getattr(message, name)
            if name == "audio_name":
                value = str(value).strip()
            elif name == "language":
                value = str(value).strip().lower()
            fields[name] = value
    for name in _TAG_FIELDS:
        if message.HasField(name):
            fields[name] = [
                MessageToDict(item, preserving_proto_field_name=True)
                for item in getattr(message, name).items
            ]
    if message.HasField("embedding") and "description" not in fields:
        fields["embedding"] = [float(value) for value in message.embedding.values]
    return fields


async def create_audios(service: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, fields in enumerate(items):
        material_id = ""
        try:
            material_id = await _create_one(service, fields)
            results.append(_ok(index, material_id))
        except Exception as exc:
            results.append(_fail(index, material_id, exc))
    await _finish(service, results, action="创建")
    return results


async def update_audios(service: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        material_id = str(item.get("id") or "").strip()
        try:
            await _update_one(service, material_id, dict(item.get("fields") or {}))
            results.append(_ok(index, material_id))
        except Exception as exc:
            results.append(_fail(index, material_id, exc))
    await _finish(service, results, action="更新")
    return results


async def delete_audios(service: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        material_id = str(item.get("id") or "").strip()
        try:
            await _delete_one(service, material_id)
            results.append(_ok(index, material_id))
        except Exception as exc:
            results.append(_fail(index, material_id, exc))
    await _finish(service, results, action="删除")
    return results


async def _create_one(service: Any, fields: dict[str, Any]) -> str:
    audio_name = str(fields.get("audio_name") or "").strip()
    if not audio_name:
        raise ValueError("audio_name 不能为空")
    language = str(fields.get("language") or "").strip().lower()
    if not language:
        raise ValueError("language 不能为空")
    now = utc_now()
    material = {**_CREATE_MATERIAL_DEFAULTS, **_pick(fields, _MATERIAL_FIELDS, _TAG_FIELDS)}
    material["created_at"] = now
    material["updated_at"] = now
    if "embedding" in fields and "description" not in fields:
        material["embedding"] = fields["embedding"]
    clear_embedding = await _apply_description_embedding(service, "", fields, material)
    if clear_embedding:
        material.pop("embedding", None)
    result = await service._materials().insert_one(material)
    material_id = str(result.inserted_id)
    saved = {**material, "_id": result.inserted_id}
    locale: dict[str, Any] | None = None
    try:
        locale = await _save_locale(
            service,
            material_id,
            {
                **_CREATE_LOCALE_DEFAULTS,
                **_pick(fields, _LOCALE_FIELDS),
                "audio_name": audio_name,
                "language": language,
            },
        )
        await _sync_after_write(
            service,
            material_id,
            saved,
            changed=set(fields) | set(_CREATE_LOCALE_DEFAULTS) | {"audio_name"},
            wrote_locale=locale,
            recompute_locale_vector=True,
            create=True,
        )
    except Exception:
        await _compensate_create_failure(service, material_id, locale)
        raise
    return material_id


async def _update_one(service: Any, material_id: str, fields: dict[str, Any]) -> None:
    oid = _parse_id(material_id)
    material_fields = _pick(fields, _MATERIAL_FIELDS, _TAG_FIELDS, ("embedding",))
    locale_fields = _normalize_locale(_pick(fields, _LOCALE_FIELDS))
    if "audio_name" in locale_fields and not str(locale_fields["audio_name"]).strip():
        raise ValueError("audio_name 不能为空")
    if not material_fields and not locale_fields:
        raise ValueError("没有要更新的字段")
    if locale_fields:
        language = str(locale_fields.get("language") or "").strip().lower()
        if not language:
            raise ValueError("language 不能为空")
        locale_fields["language"] = language
    before_material = await service._materials().find_one({"_id": oid})
    if before_material is None:
        raise ValueError(f"音频原料不存在：{material_id}")
    before_locale = None
    if locale_fields:
        before_locale = await _find_locale_snapshot(
            service, material_id, str(locale_fields["language"])
        )
    clear_embedding = await _apply_description_embedding(
        service, material_id, fields, material_fields
    )
    touch_material = bool(material_fields) or clear_embedding or "description" in fields
    material_written = False
    locale_written = False
    locale: dict[str, Any] | None = None
    saved = before_material
    try:
        if touch_material:
            material_fields["updated_at"] = utc_now()
            update: dict[str, Any] = {"$set": material_fields}
            if clear_embedding:
                update["$unset"] = {"embedding": ""}
            saved = await service._materials().find_one_and_update(
                {"_id": oid},
                update,
                return_document=ReturnDocument.AFTER,
            )
            material_written = True
        if saved is None:
            raise ValueError(f"音频原料不存在：{material_id}")
        if locale_fields:
            locale = await _save_locale(service, material_id, locale_fields)
            locale_written = True
        await _sync_after_write(
            service,
            material_id,
            saved,
            changed=set(fields) | ({"updated_at"} if touch_material else set()),
            wrote_locale=locale,
            recompute_locale_vector=_recompute_locale_vector(fields),
            touch_material=touch_material,
        )
    except Exception:
        if material_written or locale_written:
            await _compensate_update_failure(
                service,
                material_id,
                before_material=before_material,
                before_locale=before_locale,
                wrote_locale=locale,
                touched_material=material_written,
                touched_locale=locale_written,
            )
        raise


async def _delete_one(service: Any, material_id: str) -> None:
    oid = _parse_id(material_id)
    locales = _locale_collection(service)
    before_material = await service._materials().find_one({"_id": oid})
    if before_material is None:
        raise ValueError(f"音频原料不存在：{material_id}")
    rows = await locales.find({"material_id": material_id}).to_list(length=None)
    result = await service._materials().delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise ValueError(f"音频原料不存在：{material_id}")
    locales_deleted = False
    try:
        await locales.delete_many({"material_id": material_id})
        locales_deleted = True
        await _delete_index(service, material_id)
        for row in rows:
            await _delete_locale_index(service, str(row.get("_id") or ""))
    except Exception:
        await _compensate_delete_failure(
            service,
            before_material,
            rows,
            locales_deleted=locales_deleted,
        )
        raise


def _pick(fields: dict[str, Any], *groups: tuple[str, ...]) -> dict[str, Any]:
    allowed = {name for group in groups for name in group}
    return {key: value for key, value in fields.items() if key in allowed}


def _normalize_locale(fields: dict[str, Any]) -> dict[str, Any]:
    payload = dict(fields)
    if "language" in payload:
        payload["language"] = str(payload.get("language") or "").strip().lower()
    if "audio_name" in payload:
        payload["audio_name"] = str(payload.get("audio_name") or "").strip()
    for name in ("audio_url", "cover_url"):
        if name in payload:
            payload[name] = str(payload.get(name) or "").strip()
    for name in ("description", "summary"):
        if name in payload:
            payload[name] = str(payload.get(name) or "")
    return payload


def _parse_id(material_id: str) -> ObjectId:
    if not material_id or not ObjectId.is_valid(material_id):
        raise ValueError("id 不合法")
    return ObjectId(material_id)


def _recompute_locale_vector(fields: dict[str, Any]) -> bool:
    return any(name in fields for name in (*_LOCALE_TEXT_FIELDS, "language"))


async def _apply_description_embedding(
    service: Any,
    material_id: str,
    fields: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    """描述在本次写入里时，用描述原文重算原料上的 embedding。描述被清空时返回 True。"""
    if "description" not in fields:
        return False
    text = str(fields.get("description") or "").strip()
    if not text:
        target.pop("embedding", None)
        return True
    vector = await _encode_vector(service, material_id, text, required=True)
    if vector is None:
        raise RuntimeError("未配置向量模型，无法重算 embedding")
    target["embedding"] = vector
    return False


def _locale_collection(service: Any) -> Any:
    getter = getattr(service, "_locales", None)
    if callable(getter):
        return getter()
    return service._db()[service._settings.somni_mongo_material_locales_collection]


async def _find_locale_snapshot(
    service: Any, material_id: str, language: str
) -> dict[str, Any] | None:
    """与 _save_locale 相同的定位规则，供更新失败时回滚。"""
    collection = _locale_collection(service)
    current = await collection.find_one({"material_id": material_id, "language": language})
    if current is not None:
        return current
    if language:
        blank = await collection.find_one({"material_id": material_id, "language": ""})
        if blank is not None:
            return blank
    return None


async def _compensate_create_failure(
    service: Any,
    material_id: str,
    locale: dict[str, Any] | None,
) -> None:
    """创建时 ES 失败：删掉刚写入的 Mongo，并尽量清掉已写出的 ES。"""
    try:
        oid = ObjectId(material_id)
        await service._materials().delete_one({"_id": oid})
        await _locale_collection(service).delete_many({"material_id": material_id})
        await _delete_index(service, material_id)
        if locale is not None:
            await _delete_locale_index(service, str(locale.get("_id") or ""))
    except Exception as exc:
        logger.error(
            "创建补偿失败，material_id={}，原因：{}",
            material_id,
            exc,
        )


async def _compensate_update_failure(
    service: Any,
    material_id: str,
    *,
    before_material: dict[str, Any],
    before_locale: dict[str, Any] | None,
    wrote_locale: dict[str, Any] | None,
    touched_material: bool,
    touched_locale: bool,
) -> None:
    """更新时 ES 失败：Mongo 写回快照，并按快照重刷 ES。"""
    try:
        oid = _parse_id(material_id)
        if touched_material:
            restore = {key: value for key, value in before_material.items() if key != "_id"}
            await service._materials().replace_one({"_id": oid}, restore)
        if touched_locale:
            locales = _locale_collection(service)
            if before_locale is None and wrote_locale is not None:
                locale_id = wrote_locale.get("_id")
                if locale_id is not None:
                    await locales.delete_one({"_id": locale_id})
                await _delete_locale_index(service, str(locale_id or ""))
            elif before_locale is not None:
                locale_oid = before_locale["_id"]
                restore_locale = {
                    key: value for key, value in before_locale.items() if key != "_id"
                }
                await locales.replace_one({"_id": locale_oid}, restore_locale)
                source = locale_source_for_es(bson_to_jsonable(before_locale))
                if source is not None:
                    await _index_locale_document(service, str(locale_oid), source)
                else:
                    await _delete_locale_index(service, str(locale_oid))
        if touched_material:
            await _index_material_document(
                service, material_id, _material_index_body(before_material)
            )
    except Exception as exc:
        logger.error(
            "更新补偿失败，material_id={}，原因：{}",
            material_id,
            exc,
        )


async def _compensate_delete_failure(
    service: Any,
    before_material: dict[str, Any],
    locale_rows: list[dict[str, Any]],
    *,
    locales_deleted: bool,
) -> None:
    """删除失败：插回原料；仅当语言行已删时再插回语言行；并尽量重刷 ES。"""
    material_id = str(before_material.get("_id") or "")
    try:
        await service._materials().insert_one(dict(before_material))
        if locales_deleted and locale_rows:
            await _locale_collection(service).insert_many(
                [dict(row) for row in locale_rows]
            )
        if material_id:
            await _index_material_document(
                service, material_id, _material_index_body(before_material)
            )
        for row in locale_rows:
            locale_id = str(row.get("_id") or "")
            source = locale_source_for_es(bson_to_jsonable(row))
            if locale_id and source is not None:
                await _index_locale_document(service, locale_id, source)
    except Exception as exc:
        logger.error(
            "删除补偿失败，material_id={}，原因：{}",
            material_id,
            exc,
        )


async def _save_locale(
    service: Any,
    material_id: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """按 material_id + language 写入。第一次选定语言时，把空语言行改成该语言。"""
    collection = _locale_collection(service)
    language = str(fields.get("language") or "").strip().lower()
    now = utc_now()
    payload = {key: value for key, value in fields.items() if key != "language"}
    payload["language"] = language
    payload["update_time"] = now
    if language:
        current = await collection.find_one({"material_id": material_id, "language": language})
        if current is None:
            blank = await collection.find_one({"material_id": material_id, "language": ""})
            if blank is not None:
                saved = await collection.find_one_and_update(
                    {"_id": blank["_id"]},
                    {"$set": payload},
                    return_document=ReturnDocument.AFTER,
                )
                if saved is not None:
                    saved["material_id"] = material_id
                return saved
    saved = await collection.find_one_and_update(
        {"material_id": material_id, "language": language},
        {
            "$set": payload,
            "$setOnInsert": {"material_id": material_id, "create_time": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if saved is not None:
        saved["material_id"] = material_id
    return saved


async def _sync_after_write(
    service: Any,
    material_id: str,
    material: dict[str, Any],
    *,
    changed: set[str],
    wrote_locale: dict[str, Any] | None,
    recompute_locale_vector: bool,
    create: bool = False,
    touch_material: bool = True,
) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return
    if create or touch_material or any(name in changed for name in _TAG_FIELDS):
        body = _material_index_body(material)
        if create:
            await _index_material_document(service, material_id, body)
        else:
            await _patch_material_index(service, material_id, body, changed=changed)
    if wrote_locale is not None and str(wrote_locale.get("language") or "").strip():
        if create:
            source = locale_source_for_es(bson_to_jsonable(wrote_locale))
            if source is not None:
                await _index_locale_document(service, str(wrote_locale.get("_id") or ""), source)
        else:
            await _patch_locale_index(
                service,
                wrote_locale,
                changed=changed,
                recompute_vector=recompute_locale_vector,
            )


async def _patch_material_index(
    service: Any,
    material_id: str,
    body: dict[str, Any],
    *,
    changed: set[str],
) -> None:
    es_search = service._es_search
    fields = {
        key: body[key]
        for key in changed
        if key in body and key not in _ES_SKIP
    }
    if "sleep_stage_tags" in changed:
        fields["sleep_stage_names"] = body.get("sleep_stage_names") or []
    if not fields:
        # 标签等可能只改了派生字段；仍尝试整份补写缺失文档
        if any(name in changed for name in (*_MATERIAL_FIELDS, *_TAG_FIELDS, "updated_at")):
            await _index_material_document(service, material_id, body)
        return
    try:
        updated = await _patch_document(
            es_search._client,
            index=es_search.audio_index,
            doc_id=material_id,
            fields=fields,
            remove=[],
        )
    except Exception as exc:
        raise RuntimeError(f"检索索引写入失败：{exc}") from exc
    if not updated:
        await _index_material_document(service, material_id, body)


def _material_index_body(material: dict[str, Any]) -> dict[str, Any]:
    flat = bson_to_jsonable(material)
    for key in list(flat):
        if key in _ES_SKIP:
            flat.pop(key, None)
    flat["sleep_stage_names"] = extract_sleep_stage_names(flat)
    return flat


async def _index_material_document(
    service: Any,
    material_id: str,
    body: dict[str, Any],
) -> None:
    try:
        await service._es_search._client.index(
            index=service._es_search.audio_index,
            id=material_id,
            document=body,
        )
    except Exception as exc:
        raise RuntimeError(f"检索索引写入失败：{exc}") from exc


async def _patch_locale_index(
    service: Any,
    locale: dict[str, Any],
    *,
    changed: set[str],
    recompute_vector: bool,
) -> None:
    es_search = service._es_search
    if not getattr(es_search, "material_locales_index", ""):
        return
    locale_id = str(locale.get("_id") or "")
    if not locale_id:
        return
    source = locale_source_for_es(bson_to_jsonable(locale))
    if source is None:
        return
    fields = {key: source[key] for key in changed if key in source}
    language = str(source.get("language") or "")
    vector_field = "description_vector" if language == "zh" else ""
    if language == "en":
        vector_field = "description_vector_en"
    remove: list[str] = []
    if recompute_vector:
        text = build_locale_description_text(source).strip()
        fields["description_text"] = text
        if vector_field and text:
            vector = await _encode_vector(
                service,
                locale_id,
                text,
                required=False,
                language=language,
            )
            if vector is not None:
                fields[vector_field] = vector
        remove = ["description_vector", "description_vector_en"]
        if vector_field and vector_field in fields:
            remove.remove(vector_field)
    if not fields and not remove:
        return
    try:
        updated = await _patch_document(
            es_search._client,
            index=es_search.material_locales_index,
            doc_id=locale_id,
            fields=fields,
            remove=remove,
        )
    except Exception as exc:
        raise RuntimeError(f"语言索引写入失败：{exc}") from exc
    if not updated:
        await _index_locale_document(service, locale_id, source)


async def _index_locale_document(service: Any, locale_id: str, source: dict[str, Any]) -> None:
    language = str(source.get("language") or "")
    text = str(source.get("description_text") or "").strip()
    source.pop("description_vector", None)
    source.pop("description_vector_en", None)
    vector_field = "description_vector" if language == "zh" else ""
    if language == "en":
        vector_field = "description_vector_en"
    if vector_field and text:
        vector = await _encode_vector(service, locale_id, text, required=False, language=language)
        if vector is not None:
            source[vector_field] = vector
    try:
        await service._es_search._client.index(
            index=service._es_search.material_locales_index,
            id=locale_id,
            document=source,
        )
    except Exception as exc:
        raise RuntimeError(f"语言索引写入失败：{exc}") from exc


async def _patch_document(
    client: Any,
    *,
    index: str,
    doc_id: str,
    fields: dict[str, Any],
    remove: list[str],
) -> bool:
    """局部更新。文档不存在时返回 False，调用方再整份写入。"""
    try:
        if remove:
            await client.update(
                index=index,
                id=doc_id,
                script={
                    "source": (
                        "for (change in params.fields.entrySet()) {"
                        " ctx._source[change.getKey()] = change.getValue(); }"
                        " for (field in params.remove) { ctx._source.remove(field); }"
                    ),
                    "params": {"fields": fields, "remove": remove},
                },
                retry_on_conflict=3,
            )
        else:
            await client.update(
                index=index,
                id=doc_id,
                doc=fields,
                retry_on_conflict=3,
            )
    except NotFoundError:
        return False
    return True


async def _encode_vector(
    service: Any,
    doc_id: str,
    text: str,
    *,
    required: bool,
    language: str = "zh",
) -> list[float] | None:
    encoder = getattr(service, "_encoder", None)
    if encoder is None:
        if required:
            return None
        logger.warning("未配置向量模型，跳过检索向量，id={}", doc_id)
        return None
    try:
        return await encoder.encode_one(text, language=language)
    except Exception as exc:
        raise RuntimeError(f"检索向量重算失败：{exc}") from exc


async def _delete_index(service: Any, material_id: str) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return
    try:
        await es_search._client.delete(index=es_search.audio_index, id=material_id)
    except NotFoundError:
        return
    except Exception as exc:
        raise RuntimeError(f"检索索引删除失败：{exc}") from exc


async def _delete_locale_index(service: Any, locale_id: str) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is None or not locale_id or not getattr(es_search, "material_locales_index", ""):
        return
    try:
        await es_search._client.delete(index=es_search.material_locales_index, id=locale_id)
    except NotFoundError:
        return
    except Exception as exc:
        raise RuntimeError(f"语言索引删除失败：{exc}") from exc


async def _finish(service: Any, results: list[dict[str, Any]], *, action: str) -> None:
    ok_count = sum(1 for item in results if item["ok"])
    if ok_count:
        await _invalidate(service)
    if results:
        logger.info(
            "批量{}音频原料，成功={}，失败={}",
            action,
            ok_count,
            len(results) - ok_count,
        )


async def _invalidate(service: Any) -> None:
    cache = getattr(service, "_catalog_cache", None)
    if cache is not None:
        try:
            await clear_somni_audio_catalog_cache(cache._redis)
        except Exception as exc:
            logger.warning("清除量产音频目录缓存失败：{}", exc)
    retrieval = getattr(service, "_retrieval", None)
    if retrieval is not None:
        try:
            await retrieval.clear_sleep_stage_cache()
        except Exception as exc:
            logger.warning("清除量产睡眠阶段候选缓存失败：{}", exc)


def _ok(index: int, material_id: str) -> dict[str, Any]:
    return {"index": index, "ok": True, "id": material_id, "message": ""}


def _fail(index: int, material_id: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AppError):
        message = exc.message
    else:
        message = str(exc) or exc.__class__.__name__
    logger.warning("音频原料写入失败，index={}，id={}，原因：{}", index, material_id, message)
    return {"index": index, "ok": False, "id": material_id, "message": message}
