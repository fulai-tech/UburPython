"""量产音频原料批量写入。

原料表/原料索引只存状态、来源、标签和 embedding。
标题、描述、短描述、audio_url、cover_url 与检索向量写语言表/语言索引。
"""

from __future__ import annotations

from collections.abc import Callable
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
    build_tag_name_by_code_language,
    extract_sleep_stage_names,
    locale_source_for_es,
    material_content_form_tag_codes,
    material_tag_labels_from_dictionary,
)

AUDIO_WRITE_BATCH_LIMIT = 200
_LOCALE_ENCODE_BATCH = 64
_LOCALE_BULK_CHUNK = 200
_LOCALE_REFRESH_ATTEMPTS = 3
_PATCH_SCRIPT = (
    "for (change in params.fields.entrySet()) {"
    " ctx._source[change.getKey()] = change.getValue(); }"
    " for (field in params.remove) { ctx._source.remove(field); }"
)

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
    reenabling = (
        "status" in fields
        and bool(fields.get("status"))
        and not bool(before_material.get("status"))
    )
    refresh_all_locales = reenabling or any(name in fields for name in _TAG_FIELDS)
    before_locale_rows: list[dict[str, Any]] = []
    if refresh_all_locales:
        before_locale_rows = await _locale_collection(service).find(
            {"material_id": material_id}
        ).to_list(length=None)
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
            recompute_locale_vector=_recompute_locale_vector(
                fields, before_locale=before_locale
            ),
            touch_material=touch_material,
            refresh_all_locales=refresh_all_locales,
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
                before_locale_rows=before_locale_rows,
                restore_all_locales=refresh_all_locales,
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


def _recompute_locale_vector(
    fields: dict[str, Any],
    before_locale: dict[str, Any] | None = None,
) -> bool:
    # description_text 仅由 content_form 标签决定；仅当 language 实际变化时换向量字段。
    # 标签变更由 tags_changed 另行触发重算。请求里顺带带上同值 language 不重算。
    if "language" not in fields:
        return False
    new_lang = str(fields.get("language") or "").strip().lower()
    if before_locale is None:
        return bool(new_lang)
    old_lang = str(before_locale.get("language") or "").strip().lower()
    return bool(new_lang) and new_lang != old_lang


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
    before_locale_rows: list[dict[str, Any]] | None = None,
    restore_all_locales: bool = False,
) -> None:
    """更新时 ES 失败：Mongo 写回快照，并按快照重刷 ES。

    标签变更可能已刷新多条语言索引；restore_all_locales 时按变更前语言行整批回滚。
    各步骤互不阻断：词典不可用导致语言索引回滚失败时，原料索引仍要先回滚。
    """
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
            elif before_locale is not None:
                restore_locale = {
                    key: value for key, value in before_locale.items() if key != "_id"
                }
                await locales.replace_one({"_id": before_locale["_id"]}, restore_locale)
    except Exception as exc:
        logger.error("更新补偿写回 Mongo 失败，material_id={}，原因：{}", material_id, exc)

    if touched_material:
        try:
            await _index_material_document(
                service, material_id, _material_index_body(before_material)
            )
        except Exception as exc:
            logger.error("更新补偿回滚原料索引失败，material_id={}，原因：{}", material_id, exc)

    restore_rows: list[dict[str, Any]] = []
    if restore_all_locales:
        restore_rows = list(before_locale_rows or [])
    elif touched_locale and before_locale is not None:
        restore_rows = [before_locale]
    if touched_locale and before_locale is None and wrote_locale is not None:
        await _restore_locale_index(service, material_id, wrote_locale, None, before_material)
    for row in restore_rows:
        await _restore_locale_index(service, material_id, row, row, before_material)


async def _restore_locale_index(
    service: Any,
    material_id: str,
    row: dict[str, Any],
    snapshot: dict[str, Any] | None,
    before_material: dict[str, Any],
) -> None:
    """按快照回滚单条语言索引；snapshot 为 None 或无 language 时删除。"""
    locale_id = str(row.get("_id") or "")
    if not locale_id:
        return
    try:
        source = None
        if snapshot is not None and str(snapshot.get("language") or "").strip():
            source = await _locale_source_for_index(
                service, snapshot, material=before_material
            )
        if source is not None:
            await _index_locale_document(service, locale_id, source)
        else:
            await _delete_locale_index(service, locale_id)
    except Exception as exc:
        logger.error(
            "更新补偿回滚语言索引失败，material_id={}，locale_id={}，原因：{}",
            material_id,
            locale_id,
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
            source = await _locale_source_for_index(
                service, row, material=before_material
            )
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
    refresh_all_locales: bool = False,
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
    refresh_all = refresh_all_locales or any(name in changed for name in _TAG_FIELDS)
    written: dict[str, tuple[str, str]] = {}
    wrote_id = str((wrote_locale or {}).get("_id") or "")
    if wrote_locale is not None and str(wrote_locale.get("language") or "").strip():
        key: tuple[str, str] | None = None
        if create:
            source = await _locale_source_for_index(
                service, wrote_locale, material=material
            )
            if source is not None:
                key = await _index_locale_document(service, wrote_id, source)
        else:
            key = await _patch_locale_index(
                service,
                wrote_locale,
                changed=changed,
                recompute_vector=recompute_locale_vector or refresh_all,
                material=material,
            )
        if key is not None and wrote_id:
            written[wrote_id] = key
    if refresh_all and material_id and not create:
        # 标签变更或重新启用：重算所有语言的 description_text / 向量；读失败上抛以便补偿回滚。
        locale_rows = await _locale_collection(service).find(
            {"material_id": material_id}
        ).to_list(length=None)
        for row in locale_rows:
            if not str(row.get("language") or "").strip():
                continue
            row_id = str(row.get("_id") or "")
            if wrote_id and row_id == wrote_id:
                continue
            key = await _patch_locale_index(
                service,
                row,
                changed={"description_text"},
                recompute_vector=True,
                material=material,
            )
            if key is not None and row_id:
                written[row_id] = key
    await _verify_locale_search_fields(service, material_id, written)


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


async def _tag_labels_for_locale(
    service: Any,
    *,
    material: dict[str, Any] | None,
    language: str,
) -> list[str]:
    """原料 code → 内容形态词典该语言展示名。

    查询成功但无匹配名 → 空列表（可写空检索文本）。
    词典不可用（Mongo 异常）→ 抛错，避免把旧向量清成空。
    """
    codes = material_content_form_tag_codes(material)
    if not codes:
        return []
    lang = str(language or "").strip().lower() or "zh"
    return material_tag_labels_from_dictionary(
        material,
        language=lang,
        name_by_code_lang=await _content_form_names(service, codes=codes, languages=[lang]),
    )


async def _content_form_names(
    service: Any,
    *,
    codes: list[str],
    languages: list[str],
) -> dict[tuple[str, str], str]:
    """启用的内容形态词典 → (code, language) → 展示名。查询异常时抛错，不当成无标签。"""
    if not codes or not languages:
        return {}
    try:
        rows = await service._tags().find(
            {
                "code": {"$in": codes},
                "language": {"$in": languages},
                "type": "content_form",
                "status": "启用",
            },
            projection={"code": 1, "name": 1, "language": 1, "type": 1, "status": 1},
        ).to_list(length=None)
    except Exception as exc:
        raise RuntimeError(f"标签词典不可用，请重试：{exc}") from exc
    return build_tag_name_by_code_language(bson_to_jsonable(rows))


async def _locale_source_for_index(
    service: Any,
    locale: dict[str, Any],
    *,
    material: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    flat = bson_to_jsonable(locale)
    language = str(flat.get("language") or "")
    material_id = str(flat.get("material_id") or "")
    mat = material
    if mat is None and material_id:
        try:
            oid = ObjectId(material_id) if ObjectId.is_valid(material_id) else material_id
            mat = await service._materials().find_one({"_id": oid})
        except Exception as exc:
            raise RuntimeError(f"读取原料失败，请重试：{exc}") from exc
    tag_labels = await _tag_labels_for_locale(service, material=mat, language=language)
    return locale_source_for_es(flat, tag_labels=tag_labels)


async def _patch_locale_index(
    service: Any,
    locale: dict[str, Any],
    *,
    changed: set[str],
    recompute_vector: bool,
    material: dict[str, Any] | None = None,
) -> tuple[str, str] | None:
    """返回写入的 (language, description_text)；没写检索文本时返回 None。"""
    es_search = service._es_search
    if not getattr(es_search, "material_locales_index", ""):
        return None
    locale_id = str(locale.get("_id") or "")
    if not locale_id:
        return None
    source = await _locale_source_for_index(service, locale, material=material)
    if source is None:
        return None
    fields = {key: source[key] for key in changed if key in source}
    language = str(source.get("language") or "")
    vector_field = "description_vector" if language == "zh" else ""
    if language == "en":
        vector_field = "description_vector_en"
    remove: list[str] = []
    if recompute_vector:
        text = str(source.get("description_text") or "").strip()
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
        return None
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
        return await _index_locale_document(service, locale_id, source)
    return _search_key(source) if recompute_vector else None


async def _index_locale_document(
    service: Any, locale_id: str, source: dict[str, Any]
) -> tuple[str, str]:
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
    return _search_key(source)


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
                    "source": _PATCH_SCRIPT,
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


async def refresh_locale_indexes_for_materials(
    service: Any,
    materials: list[dict[str, Any]],
    *,
    languages: set[str] | None = None,
) -> int:
    """按原料当前 content_form 标签批量刷新语言索引的检索文本与向量，返回写入条数。

    只用传入原料的 id，素材标签、语言行、词典都在写入前后从 Mongo 重读。
    languages 非空时只刷指定语言；用于词典展示名变更后的关联刷新。
    """
    es_search = getattr(service, "_es_search", None)
    if es_search is None or not getattr(es_search, "material_locales_index", ""):
        return 0
    material_ids = sorted(
        {str(material.get("_id") or "").strip() for material in materials} - {""}
    )
    if not material_ids:
        return 0
    wanted = {str(lang).strip().lower() for lang in (languages or set()) if str(lang).strip()}
    return await _converge_locale_search_fields(
        service,
        material_ids,
        include=lambda row: not wanted or _row_language(row) in wanted,
        written={},
    )


async def _converge_locale_search_fields(
    service: Any,
    material_ids: list[str],
    *,
    include: Callable[[dict[str, Any]], bool],
    written: dict[str, tuple[str, str]],
) -> int:
    """写后复查：按 Mongo 最新数据重算检索文本，和已写入 ES 的不一致就重写，直到一致。

    读数据到写 ES 之间可能有别的请求改了素材标签、语言行或词典；每个写入方写完都复查一次，
    较早的写入即使后到，也会被复查发现并改回，不依赖进程内锁，多实例同样成立。
    written：locale_id → 已写入的 (language, description_text)；这些行始终参与复查。
    """
    written = dict(written)
    count = 0
    for attempt in range(_LOCALE_REFRESH_ATTEMPTS + 1):
        expected = await _expected_locale_sources(
            service,
            material_ids,
            include=lambda row: include(row) or str(row.get("_id") or "") in written,
        )
        stale = [
            (locale_id, source)
            for locale_id, source in expected.items()
            if written.get(locale_id) != _search_key(source)
        ]
        gone = [locale_id for locale_id in written if locale_id not in expected]
        if not stale and not gone:
            return count
        if attempt == _LOCALE_REFRESH_ATTEMPTS:
            raise RuntimeError("素材标签或词典在刷新期间持续变化，请重试")
        await _attach_locale_vectors(service, stale)
        await _bulk_update_locale_search_fields(service, stale)
        for locale_id in gone:
            await _delete_locale_index(service, locale_id)
            written.pop(locale_id, None)
        for locale_id, source in stale:
            written[locale_id] = _search_key(source)
        count += len(stale)
    return count


async def _expected_locale_sources(
    service: Any,
    material_ids: list[str],
    *,
    include: Callable[[dict[str, Any]], bool],
) -> dict[str, dict[str, Any]]:
    """当前 Mongo 状态下各语言行应有的 ES _source；语言行已删或无 language 的不在结果里。"""
    rows = await _locale_collection(service).find(
        {"material_id": {"$in": material_ids}}
    ).to_list(length=None)
    targets = [
        bson_to_jsonable(row) for row in rows if _row_language(row) and include(row)
    ]
    if not targets:
        return {}
    try:
        materials = await service._materials().find(
            {"_id": {"$in": [_object_id(material_id) for material_id in material_ids]}},
            projection={"content_form_tags": 1},
        ).to_list(length=None)
    except Exception as exc:
        raise RuntimeError(f"读取原料失败，请重试：{exc}") from exc
    by_id = {str(material.get("_id") or ""): material for material in materials}
    codes = sorted(
        {
            code
            for row in targets
            for code in material_content_form_tag_codes(by_id.get(str(row.get("material_id"))))
        }
    )
    name_by = await _content_form_names(
        service, codes=codes, languages=sorted({_row_language(row) for row in targets})
    )
    expected: dict[str, dict[str, Any]] = {}
    for row in targets:
        labels = material_tag_labels_from_dictionary(
            by_id.get(str(row.get("material_id") or "")),
            language=_row_language(row),
            name_by_code_lang=name_by,
        )
        source = locale_source_for_es(row, tag_labels=labels)
        if source is not None:
            expected[str(row["_id"])] = source
    return expected


def _row_language(row: dict[str, Any]) -> str:
    return str(row.get("language") or "").strip().lower()


def _object_id(value: str) -> Any:
    return ObjectId(value) if ObjectId.is_valid(value) else value


def _search_key(source: dict[str, Any]) -> tuple[str, str]:
    return (
        str(source.get("language") or ""),
        str(source.get("description_text") or "").strip(),
    )


async def _verify_locale_search_fields(
    service: Any, material_id: str, written: dict[str, tuple[str, str]]
) -> None:
    """单条写入后的复查；Mongo 已提交，复查失败只记日志，不回滚本次写入。"""
    if not written or not material_id:
        return
    try:
        await _converge_locale_search_fields(
            service, [material_id], include=lambda _row: False, written=written
        )
    except Exception as exc:
        logger.error("语言索引写后复查失败，material_id={}，原因：{}", material_id, exc)


async def _attach_locale_vectors(
    service: Any, sources: list[tuple[str, dict[str, Any]]]
) -> None:
    """按语言分组、文本去重后批量编码，写回各 source 的向量字段。"""
    pending: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for _, source in sources:
        source.pop("description_vector", None)
        source.pop("description_vector_en", None)
        language = str(source.get("language") or "")
        text = str(source.get("description_text") or "").strip()
        if language in ("zh", "en") and text:
            pending.setdefault(language, {}).setdefault(text, []).append(source)
    if not pending:
        return
    encoder = getattr(service, "_encoder", None)
    if encoder is None:
        logger.warning("未配置向量模型，批量刷新语言索引跳过检索向量")
        return
    for language, by_text in pending.items():
        field = "description_vector" if language == "zh" else "description_vector_en"
        texts = list(by_text)
        for start in range(0, len(texts), _LOCALE_ENCODE_BATCH):
            chunk = texts[start : start + _LOCALE_ENCODE_BATCH]
            try:
                vectors = await encoder.encode(chunk, language=language)
            except Exception as exc:
                raise RuntimeError(f"检索向量重算失败：{exc}") from exc
            for text, vector in zip(chunk, vectors, strict=True):
                for source in by_text[text]:
                    source[field] = vector


async def _bulk_update_locale_search_fields(
    service: Any, sources: list[tuple[str, dict[str, Any]]]
) -> None:
    """只更新 description_text 与向量，不覆盖标题等并发可能已改的字段；文档缺失时写入整份。"""
    es_search = service._es_search
    index = es_search.material_locales_index
    for start in range(0, len(sources), _LOCALE_BULK_CHUNK):
        operations: list[dict[str, Any]] = []
        for locale_id, source in sources[start : start + _LOCALE_BULK_CHUNK]:
            fields = {"description_text": source.get("description_text", "")}
            remove = ["description_vector", "description_vector_en"]
            for field in ("description_vector", "description_vector_en"):
                if field in source:
                    fields[field] = source[field]
                    remove.remove(field)
            operations.append(
                {"update": {"_index": index, "_id": locale_id, "retry_on_conflict": 3}}
            )
            operations.append(
                {
                    "script": {
                        "source": _PATCH_SCRIPT,
                        "params": {"fields": fields, "remove": remove},
                    },
                    "upsert": source,
                }
            )
        try:
            response = await es_search._client.bulk(operations=operations)
        except Exception as exc:
            raise RuntimeError(f"语言索引批量写入失败：{exc}") from exc
        body = getattr(response, "body", response)
        if body.get("errors"):
            reasons = [
                item.get("update", {}).get("error")
                for item in body.get("items") or []
                if item.get("update", {}).get("error")
            ]
            raise RuntimeError(f"语言索引批量写入失败：{reasons[:1]}")


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
    es_search = getattr(service, "_es_search", None)
    if es_search is not None and hasattr(es_search, "refresh_audio_index"):
        try:
            await es_search.refresh_audio_index()
        except Exception as exc:
            # refresh 失败时保留旧缓存，避免立刻把未可见的旧 ES 结果写回无 TTL 缓存。
            logger.warning("写入后刷新 ES 索引失败，跳过清缓存：{}", exc)
            return
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
