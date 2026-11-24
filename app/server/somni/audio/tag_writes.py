"""量产标签字典批量写入：只写 somni_audio_tag_dictionary 与标签索引。"""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from elasticsearch import NotFoundError
from loguru import logger
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.cache.somni_audio_catalog_cache import clear_somni_audio_catalog_cache
from app.core.bson_util import bson_to_jsonable, utc_now
from app.core.exceptions import AppError
from app.server.somni.audio.writes import refresh_locale_indexes_for_materials

_FIELDS = (
    "type",
    "code",
    "language",
    "name",
    "status",
    "parent_id",
    "created_by",
    "updated_by",
)
_REQUIRED = ("type", "code", "language", "name")
_VECTOR_FIELDS = ("name", "language")
_CONTENT_FORM = "content_form"
_LOCALE_AFFECTING_FIELDS = ("name", "language", "type", "code", "status")
_CREATE_DEFAULTS: dict[str, Any] = {
    "status": "启用",
    "parent_id": None,
    "created_by": "",
    "updated_by": "",
}


def audio_tag_input_to_fields(message: Any) -> dict[str, Any]:
    """只取出调用方显式设置的字段。"""
    fields: dict[str, Any] = {}
    for name in _FIELDS:
        if message.HasField(name):
            fields[name] = getattr(message, name)
    return _normalize(fields)


async def create_audio_tags(service: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, fields in enumerate(items):
        tag_id = ""
        try:
            tag_id = await _create_one(service, fields)
            results.append(_ok(index, tag_id))
        except Exception as exc:
            results.append(_fail(index, tag_id, exc))
    await _finish(service, results, action="创建")
    return results


async def update_audio_tags(service: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        tag_id = str(item.get("id") or "").strip()
        try:
            await _update_one(service, tag_id, dict(item.get("fields") or {}))
            results.append(_ok(index, tag_id))
        except Exception as exc:
            results.append(_fail(index, tag_id, exc))
    await _finish(service, results, action="更新")
    return results


async def delete_audio_tags(service: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        tag_id = str(item.get("id") or "").strip()
        try:
            await _delete_one(service, tag_id)
            results.append(_ok(index, tag_id))
        except Exception as exc:
            results.append(_fail(index, tag_id, exc))
    await _finish(service, results, action="删除")
    return results


async def _create_one(service: Any, fields: dict[str, Any]) -> str:
    payload = _normalize(fields)
    _require_identity(payload)
    now = utc_now()
    doc = {**_CREATE_DEFAULTS, **payload}
    doc["created_at"] = now
    doc["updated_at"] = now
    try:
        result = await service._tags().insert_one(doc)
    except DuplicateKeyError as exc:
        raise ValueError("标签已存在：type + code + language") from exc
    tag_id = str(result.inserted_id)
    saved = {**doc, "_id": result.inserted_id}
    try:
        await _index_tag(service, tag_id, saved)
        await _refresh_linked_material_locales(service, saved)
    except Exception:
        await _compensate_create_failure(service, tag_id)
        try:
            await _refresh_linked_material_locales(service, saved)
        except Exception as restore_exc:
            logger.error(
                "标签创建补偿后语言索引回滚失败，tag_id={}，原因：{}",
                tag_id,
                restore_exc,
            )
        raise
    return tag_id


async def _update_one(service: Any, tag_id: str, fields: dict[str, Any]) -> None:
    oid = _parse_id(tag_id)
    payload = _normalize(fields)
    _reject_blank_identity(payload)
    if not payload:
        raise ValueError("没有要更新的字段")
    before = await service._tags().find_one({"_id": oid})
    if before is None:
        raise ValueError(f"标签不存在：{tag_id}")
    recompute_vector = any(name in payload for name in _VECTOR_FIELDS)
    refresh_locales = _should_refresh_linked_locales(before, payload)
    payload["updated_at"] = utc_now()
    try:
        saved = await service._tags().find_one_and_update(
            {"_id": oid},
            {"$set": payload},
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError as exc:
        raise ValueError("标签已存在：type + code + language") from exc
    if saved is None:
        raise ValueError(f"标签不存在：{tag_id}")
    try:
        await _patch_tag(
            service,
            tag_id,
            saved,
            changed=set(payload),
            recompute_vector=recompute_vector,
        )
        if refresh_locales:
            for tag in _locale_refresh_keys(before, saved):
                await _refresh_linked_material_locales(service, tag)
    except Exception:
        await _compensate_update_failure(service, tag_id, before=before)
        if refresh_locales:
            # 新旧两个键都可能已被刷过，回滚后按恢复的词典各重刷一次。
            for tag in _locale_refresh_keys(before, saved):
                try:
                    await _refresh_linked_material_locales(service, tag)
                except Exception as restore_exc:
                    logger.error(
                        "标签更新后语言索引回滚失败，tag_id={}，原因：{}",
                        tag_id,
                        restore_exc,
                    )
        raise


async def _delete_one(service: Any, tag_id: str) -> None:
    oid = _parse_id(tag_id)
    before = await service._tags().find_one({"_id": oid})
    if before is None:
        raise ValueError(f"标签不存在：{tag_id}")
    result = await service._tags().delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise ValueError(f"标签不存在：{tag_id}")
    try:
        await _delete_index(service, tag_id)
        await _refresh_linked_material_locales(service, before)
    except Exception:
        await _compensate_delete_failure(service, before)
        try:
            await _refresh_linked_material_locales(service, before)
        except Exception as restore_exc:
            logger.error(
                "标签删除补偿后语言索引回滚失败，tag_id={}，原因：{}",
                tag_id,
                restore_exc,
            )
        raise


def _is_content_form_tag(doc: dict[str, Any] | None) -> bool:
    if not doc:
        return False
    return str(doc.get("type") or "").strip() == _CONTENT_FORM


def _should_refresh_linked_locales(
    before: dict[str, Any], payload: dict[str, Any]
) -> bool:
    """内容形态标签的展示名/语言/code/type/status 变更时，需重刷关联素材语言索引。"""
    if not any(name in payload for name in _LOCALE_AFFECTING_FIELDS):
        return False
    after_type = str(payload.get("type") or before.get("type") or "").strip()
    return _is_content_form_tag(before) or after_type == _CONTENT_FORM


def _locale_refresh_keys(
    before: dict[str, Any], saved: dict[str, Any]
) -> list[dict[str, Any]]:
    """改名只刷一次；改 type/code/language 时旧键、新键各刷一次。"""
    def key(tag: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(tag.get("type") or "").strip(),
            str(tag.get("code") or "").strip(),
            str(tag.get("language") or "").strip().lower(),
        )

    if key(before) == key(saved):
        return [saved]
    return [before, saved]


async def _refresh_linked_material_locales(service: Any, tag: dict[str, Any]) -> None:
    """重刷挂了该 content_form code 的原料语言索引。"""
    if not _is_content_form_tag(tag):
        return
    code = str(tag.get("code") or "").strip()
    if not code:
        return
    language = str(tag.get("language") or "").strip().lower()
    materials_getter = getattr(service, "_materials", None)
    if not callable(materials_getter):
        logger.warning("服务未暴露 _materials，跳过标签关联语言索引刷新 code={}", code)
        return
    try:
        materials = await materials_getter().find(
            {"content_form_tags.code": code}
        ).to_list(length=None)
    except Exception as exc:
        raise RuntimeError(f"查询标签关联原料失败：{exc}") from exc
    await refresh_locale_indexes_for_materials(
        service,
        materials,
        languages={language} if language else None,
    )


def _normalize(fields: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in _FIELDS:
        if name not in fields:
            continue
        if name == "parent_id":
            parent = str(fields.get(name) or "").strip()
            payload[name] = parent or None
            continue
        value = str(fields.get(name) or "").strip()
        if name == "language":
            value = value.lower()
        payload[name] = value
    return payload


def _require_identity(fields: dict[str, Any]) -> None:
    missing = [name for name in _REQUIRED if not str(fields.get(name) or "").strip()]
    if missing:
        raise ValueError(f"{'、'.join(missing)} 不能为空")


def _reject_blank_identity(fields: dict[str, Any]) -> None:
    blank = [name for name in _REQUIRED if name in fields and not str(fields.get(name) or "").strip()]
    if blank:
        raise ValueError(f"{'、'.join(blank)} 不能为空")


def _parse_id(tag_id: str) -> ObjectId:
    if not tag_id or not ObjectId.is_valid(tag_id):
        raise ValueError("id 不合法")
    return ObjectId(tag_id)


def _tag_source(doc: dict[str, Any]) -> dict[str, Any]:
    payload = bson_to_jsonable(doc)
    for key in ("_id", "id", "name_en", "name_vector", "name_en_vector"):
        payload.pop(key, None)
    parent = payload.pop("parent_id", None)
    payload.pop("parent_tag_id", None)
    payload["parent_id"] = None if parent in (None, "") else str(parent)
    language = str(payload.get("language") or "").strip().lower()
    payload["language"] = language or "zh"
    return payload


async def _index_tag(service: Any, tag_id: str, doc: dict[str, Any]) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return
    payload = _tag_source(doc)
    _apply_tag_vector(payload, await _vector_for(service, tag_id, payload))
    try:
        await es_search._client.index(
            index=es_search.tag_dictionary_index,
            id=tag_id,
            document=payload,
        )
    except Exception as exc:
        raise RuntimeError(f"标签索引写入失败：{exc}") from exc


async def _patch_tag(
    service: Any,
    tag_id: str,
    doc: dict[str, Any],
    *,
    changed: set[str],
    recompute_vector: bool,
) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return
    flat = _tag_source(doc)
    fields = {key: flat[key] for key in changed if key in flat}
    remove: list[str] = []
    if recompute_vector:
        vector_field, vector = await _vector_for(service, tag_id, flat)
        if vector_field and vector is not None:
            fields[vector_field] = vector
        remove = ["name_vector", "name_en_vector"]
        if vector_field:
            remove.remove(vector_field)
    try:
        updated = await _patch_document(
            es_search._client,
            index=es_search.tag_dictionary_index,
            doc_id=tag_id,
            fields=fields,
            remove=remove,
        )
    except Exception as exc:
        raise RuntimeError(f"标签索引写入失败：{exc}") from exc
    if not updated:
        await _index_tag(service, tag_id, doc)


def _apply_tag_vector(payload: dict[str, Any], vector: tuple[str, list[float] | None]) -> None:
    field, values = vector
    payload.pop("name_vector", None)
    payload.pop("name_en_vector", None)
    if field and values is not None:
        payload[field] = values


async def _vector_for(
    service: Any,
    tag_id: str,
    doc: dict[str, Any],
) -> tuple[str, list[float] | None]:
    language = str(doc.get("language") or "")
    name = str(doc.get("name") or "").strip()
    if language == "zh" and name:
        return "name_vector", await _encode_name(service, tag_id, name, "zh")
    if language == "en" and name:
        return "name_en_vector", await _encode_name(service, tag_id, name, "en")
    return "", None


async def _encode_name(service: Any, tag_id: str, name: str, language: str) -> list[float]:
    encoder = getattr(service, "_encoder", None)
    if encoder is None:
        raise RuntimeError("未配置向量模型，无法重算标签向量")
    try:
        return await encoder.encode_one(name, language=language)
    except Exception as exc:
        field = "name_vector" if language == "zh" else "name_en_vector"
        raise RuntimeError(f"{field} 重算失败：{exc}") from exc


async def _patch_document(
    client: Any,
    *,
    index: str,
    doc_id: str,
    fields: dict[str, Any],
    remove: list[str],
) -> bool:
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


async def _delete_index(service: Any, tag_id: str) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return
    try:
        await es_search._client.delete(index=es_search.tag_dictionary_index, id=tag_id)
    except NotFoundError:
        return
    except Exception as exc:
        raise RuntimeError(f"标签索引删除失败：{exc}") from exc


async def _compensate_create_failure(service: Any, tag_id: str) -> None:
    try:
        await service._tags().delete_one({"_id": ObjectId(tag_id)})
        await _delete_index(service, tag_id)
    except Exception as exc:
        logger.error("标签创建补偿失败，tag_id={}，原因：{}", tag_id, exc)


async def _compensate_update_failure(
    service: Any, tag_id: str, *, before: dict[str, Any]
) -> None:
    try:
        oid = ObjectId(tag_id)
        restore = {key: value for key, value in before.items() if key != "_id"}
        await service._tags().replace_one({"_id": oid}, restore)
        await _index_tag(service, tag_id, before)
    except Exception as exc:
        logger.error("标签更新补偿失败，tag_id={}，原因：{}", tag_id, exc)


async def _compensate_delete_failure(service: Any, before: dict[str, Any]) -> None:
    tag_id = str(before.get("_id") or "")
    try:
        await service._tags().insert_one(dict(before))
        if tag_id:
            await _index_tag(service, tag_id, before)
    except Exception as exc:
        logger.error("标签删除补偿失败，tag_id={}，原因：{}", tag_id, exc)


async def _finish(service: Any, results: list[dict[str, Any]], *, action: str) -> None:
    ok_count = sum(1 for item in results if item["ok"])
    if ok_count:
        await _invalidate(service)
    if results:
        logger.info("批量{}标签，成功={}，失败={}", action, ok_count, len(results) - ok_count)


async def _invalidate(service: Any) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is not None and hasattr(es_search, "refresh_audio_index"):
        try:
            await es_search.refresh_audio_index()
        except Exception as exc:
            # refresh 失败时保留进程内标签缓存与目录缓存，避免读到未可见的旧索引。
            logger.warning("标签写入后刷新 ES 索引失败，跳过清缓存：{}", exc)
            return
    service._content_form_tag_bank_cache = None
    service._content_form_tag_labels_cache = None
    if es_search is not None and hasattr(es_search, "clear_content_tag_vectors_cache"):
        es_search.clear_content_tag_vectors_cache()
    cache = getattr(service, "_catalog_cache", None)
    if cache is not None:
        try:
            await clear_somni_audio_catalog_cache(cache._redis)
        except Exception as exc:
            logger.warning("清除量产音频目录缓存失败：{}", exc)


def _ok(index: int, tag_id: str) -> dict[str, Any]:
    return {"index": index, "ok": True, "id": tag_id, "message": ""}


def _fail(index: int, tag_id: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AppError):
        message = exc.message
    else:
        message = str(exc) or exc.__class__.__name__
    logger.warning("标签写入失败，index={}，id={}，原因：{}", index, tag_id, message)
    return {"index": index, "ok": False, "id": tag_id, "message": message}
