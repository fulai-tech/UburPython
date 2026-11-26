"""量产标签字典批量写入：只写 somni_audio_tag_dictionary 与标签索引。"""

from __future__ import annotations

import asyncio
from typing import Any

from bson import ObjectId
from elasticsearch import NotFoundError
from loguru import logger
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.cache.somni_audio_catalog_cache import clear_somni_audio_catalog_cache
from app.core.bson_util import bson_to_jsonable, utc_now
from app.core.exceptions import AppError
from app.server.somni.audio.writes import refresh_locales_for_content_codes

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
_LABEL_FIELDS = ("name", "status", "language", "code", "type")
_LABEL_TOUCH = ("name", "status", "language", "type")
_CREATE_DEFAULTS: dict[str, Any] = {
    "status": "启用",
    "parent_id": None,
    "created_by": "",
    "updated_by": "",
    "version": 0,
}


class TagCreateUncertainError(RuntimeError):
    """创建失败且补偿未完成；tag_id 对调用方仍有核对价值。"""

    def __init__(self, tag_id: str, cause: Exception) -> None:
        super().__init__(str(cause) or cause.__class__.__name__)
        self.tag_id = tag_id


class TagVersionConflictError(ValueError):
    """读到的标签版本已过期，别的请求先写入了；调用方应刷新后重试。"""


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
            if isinstance(exc, TagCreateUncertainError):
                tag_id = exc.tag_id
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
    if _is_content_form(doc):
        doc["locale_refresh_pending"] = True
    tag_oid = ObjectId()
    doc["_id"] = tag_oid
    try:
        result = await service._tags().insert_one(doc)
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_create_failure(service, str(tag_oid), doc)
        raise
    except DuplicateKeyError as exc:
        raise ValueError("标签已存在：type + code + language") from exc
    except Exception as exc:
        compensated = await _compensate_create_failure(service, str(tag_oid), doc)
        if not compensated:
            raise TagCreateUncertainError(str(tag_oid), exc) from exc
        raise
    tag_id = str(result.inserted_id)
    saved = {**doc, "_id": result.inserted_id}
    try:
        await _index_tag(service, tag_id, saved)
        await _refresh_content_labels(service, saved)
        await _clear_locale_refresh_pending(service, result.inserted_id, saved)
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_create_failure(service, tag_id, saved)
        raise
    except Exception as exc:
        compensated = await _compensate_create_failure(service, tag_id, saved)
        if not compensated:
            raise TagCreateUncertainError(tag_id, exc) from exc
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
    if "code" in payload and payload["code"] != str(before.get("code") or "").strip():
        raise ValueError("code 不能修改")
    payload.pop("code", None)
    if not payload and not before.get("locale_refresh_pending"):
        return
    if _should_refresh_labels(before, payload):
        payload["locale_refresh_pending"] = True
    if not payload:
        return
    recompute_vector = any(name in payload for name in _VECTOR_FIELDS)
    payload["updated_at"] = utc_now()
    # 版本号 CAS：只有库里仍是读到的那一版才写入，并把版本 +1。
    # 老数据没有 version 字段时 filter 写 None，可同时匹配缺失和 null。
    base_version = before.get("version")
    written_version = int(base_version or 0) + 1
    payload["version"] = written_version
    written_fields = _own_fields(payload)
    saved = before
    try:
        saved = await service._tags().find_one_and_update(
            {"_id": oid, "version": base_version},
            {"$set": payload},
            return_document=ReturnDocument.AFTER,
        )
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_update_failure(
            service,
            tag_id,
            before=before,
            after=saved,
            written_version=written_version,
            written_fields=written_fields,
        )
        raise
    except DuplicateKeyError as exc:
        raise ValueError("标签已存在：type + code + language") from exc
    except Exception:
        await _compensate_update_failure(
            service,
            tag_id,
            before=before,
            after=saved,
            written_version=written_version,
            written_fields=written_fields,
        )
        raise
    if saved is None:
        raise TagVersionConflictError(f"标签已被其他请求修改，请刷新后重试：{tag_id}")
    try:
        await _patch_tag(
            service,
            tag_id,
            saved,
            changed=set(payload) - {"locale_refresh_pending"},
            recompute_vector=recompute_vector,
        )
        if _content_label_refresh_due(before, saved):
            await _refresh_content_labels(service, before, saved)
            await _clear_locale_refresh_pending(service, oid, saved)
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_update_failure(
            service,
            tag_id,
            before=before,
            after=saved,
            written_version=written_version,
            written_fields=written_fields,
        )
        raise
    except Exception:
        await _compensate_update_failure(
            service,
            tag_id,
            before=before,
            after=saved,
            written_version=written_version,
            written_fields=written_fields,
        )
        raise


async def _delete_one(service: Any, tag_id: str) -> None:
    oid = _parse_id(tag_id)
    before = await service._tags().find_one({"_id": oid})
    if before is None:
        raise ValueError(f"标签不存在：{tag_id}")
    try:
        result = await service._tags().delete_one({"_id": oid})
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_delete_failure(service, before)
        raise
    except Exception:
        await _compensate_delete_failure(service, before)
        raise
    if result.deleted_count == 0:
        raise ValueError(f"标签不存在：{tag_id}")
    try:
        await _delete_index(service, tag_id)
        await _refresh_content_labels(service, before)
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_delete_failure(service, before)
        raise
    except Exception:
        await _compensate_delete_failure(service, before)
        raise


def _content_codes(doc: dict[str, Any] | None) -> set[str]:
    if not doc or str(doc.get("type") or "").strip() != "content_form":
        return set()
    code = str(doc.get("code") or "").strip()
    return {code} if code else set()


def _content_ids(doc: dict[str, Any] | None) -> set[str]:
    if not _is_content_form(doc):
        return set()
    tag_id = str((doc or {}).get("_id") or "").strip()
    return {tag_id} if tag_id else set()


def _label_inputs_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return any(before.get(name) != after.get(name) for name in _LABEL_FIELDS)


def _is_content_form(doc: dict[str, Any] | None) -> bool:
    return str((doc or {}).get("type") or "").strip() == "content_form"


def _labels_touched(before: dict[str, Any], payload: dict[str, Any]) -> bool:
    return any(
        name in payload and payload[name] != before.get(name) for name in _LABEL_TOUCH
    )


def _content_label_refresh_due(before: dict[str, Any], saved: dict[str, Any]) -> bool:
    if not (_is_content_form(before) or _is_content_form(saved)):
        return False
    return bool(saved.get("locale_refresh_pending")) or _label_inputs_changed(before, saved)


def _should_refresh_labels(before: dict[str, Any], payload: dict[str, Any]) -> bool:
    """内容形态标签的名字或启停还没刷完，或者这次又改了检索会用到的字段。"""
    merged = {**before, **payload}
    if not (_is_content_form(before) or _is_content_form(merged)):
        return False
    if before.get("locale_refresh_pending"):
        return True
    return _labels_touched(before, payload)


async def _clear_locale_refresh_pending(
    service: Any, tag_oid: Any, doc: dict[str, Any]
) -> None:
    if not doc.get("locale_refresh_pending") and not _content_codes(doc):
        return
    if not _is_content_form(doc) and not doc.get("locale_refresh_pending"):
        return
    await service._tags().update_one(
        {"_id": tag_oid},
        {"$set": {"locale_refresh_pending": False}},
    )


async def _refresh_content_labels(service: Any, *docs: dict[str, Any]) -> None:
    codes: set[str] = set()
    tag_ids: set[str] = set()
    for doc in docs:
        codes.update(_content_codes(doc))
        tag_ids.update(_content_ids(doc))
    await refresh_locales_for_content_codes(service, codes, tag_ids)


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
    blank = [
        name
        for name in _REQUIRED
        if name in fields and not str(fields.get(name) or "").strip()
    ]
    if blank:
        raise ValueError(f"{'、'.join(blank)} 不能为空")


def _parse_id(tag_id: str) -> ObjectId:
    if not tag_id or not ObjectId.is_valid(tag_id):
        raise ValueError("id 不合法")
    return ObjectId(tag_id)


def _clear_cancellation() -> None:
    """先清除当前任务的取消状态，确保补偿 await 能执行完。"""
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()


def _tag_source(doc: dict[str, Any]) -> dict[str, Any]:
    payload = bson_to_jsonable(doc)
    for key in ("_id", "id", "name_en", "name_vector", "name_en_vector", "locale_refresh_pending"):
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
            guard=("version", int(doc.get("version") or 0)),
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
    guard: tuple[str, int] | None = None,
) -> bool:
    """局部更新。guard=(字段名, 版本号)：ES 上该字段已更大时整次更新 noop，防止旧版本盖掉新版本。"""
    try:
        if remove or guard is not None:
            body = (
                "for (change in params.fields.entrySet()) {"
                " ctx._source[change.getKey()] = change.getValue(); }"
                " for (field in params.remove) { ctx._source.remove(field); }"
            )
            params: dict[str, Any] = {"fields": fields, "remove": remove}
            if guard is not None:
                params["guard_field"], params["guard_value"] = guard
                body = (
                    "if (ctx._source[params.guard_field] != null"
                    " && ctx._source[params.guard_field] > params.guard_value) {"
                    " ctx.op = 'noop'; } else { "
                    + body
                    + " ctx._source[params.guard_field] = params.guard_value; }"
                )
            await client.update(
                index=index,
                id=doc_id,
                script={"source": body, "params": params},
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


async def _compensate_create_failure(
    service: Any, tag_id: str, doc: dict[str, Any]
) -> bool:
    try:
        await service._tags().delete_one({"_id": ObjectId(tag_id)})
        await _delete_index(service, tag_id)
        await refresh_locales_for_content_codes(
            service,
            _content_codes(doc),
            _content_ids(doc),
        )
        return True
    except Exception as exc:
        logger.error("标签创建补偿失败，tag_id={}，原因：{}", tag_id, exc)
        return False


_FLAG_FIELDS = ("updated_at", "version", "locale_refresh_pending")


def _own_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """本次请求真正要写的业务字段，用于版本已被推进时只撤销这些字段。"""
    return {key: value for key, value in payload.items() if key not in _FLAG_FIELDS}


async def _compensate_update_failure(
    service: Any,
    tag_id: str,
    *,
    before: dict[str, Any],
    after: dict[str, Any] | None,
    written_version: int,
    written_fields: dict[str, Any],
) -> None:
    """更新失败的补偿。

    - 库里仍是本次写入的版本：整条回滚到 before，版本号 +1，让旧版本的写入再也匹配不上。
    - 版本已被别的请求推进：整条回滚会抹掉对方的修改，只撤销本次改过、且仍是本次写入值的字段。
    索引与检索文本最后都以数据库当前记录为准，不用快照。
    """
    try:
        oid = ObjectId(tag_id)
        tags = service._tags()
        codes = _content_codes(before) | _content_codes(after or {})
        result = await tags.replace_one(
            {"_id": oid, "version": written_version},
            _restore_doc(before, written_version + 1, codes),
        )
        whole_restored = bool(getattr(result, "matched_count", 1))
        if not whole_restored:
            await _revert_own_fields(tags, oid, before, written_fields)
        current = await tags.find_one({"_id": oid})
        if current is not None:
            await _index_tag(service, tag_id, current)
        await refresh_locales_for_content_codes(
            service,
            codes,
            _content_ids(current) | _content_ids(after or {}),
        )
        if codes and whole_restored:
            await tags.update_one(
                {"_id": oid, "version": written_version + 1},
                {"$set": {"locale_refresh_pending": False}},
            )
    except Exception as exc:
        logger.error("标签更新补偿失败，tag_id={}，原因：{}", tag_id, exc)


def _restore_doc(before: dict[str, Any], version: int, codes: set[str]) -> dict[str, Any]:
    restore = {key: value for key, value in before.items() if key not in ("_id", "version")}
    restore["version"] = version
    if codes:
        restore["locale_refresh_pending"] = True
    return restore


async def _revert_own_fields(
    tags: Any,
    oid: ObjectId,
    before: dict[str, Any],
    written_fields: dict[str, Any],
) -> None:
    """逐字段撤销：只有字段当前仍是本次写入的值时才改回原值，别人改过的字段保持不动。"""
    for name, written_value in written_fields.items():
        await tags.update_one(
            {"_id": oid, name: written_value},
            {"$set": {name: before.get(name)}},
        )


async def _compensate_delete_failure(service: Any, before: dict[str, Any]) -> None:
    tag_id = str(before.get("_id") or "")
    try:
        tag_oid = before.get("_id")
        restored = {key: value for key, value in before.items() if key != "_id"}
        codes = _content_codes(before)
        if codes:
            restored["locale_refresh_pending"] = True
        await service._tags().replace_one(
            {"_id": tag_oid},
            restored,
            upsert=True,
        )
        if tag_id:
            await _index_tag(service, tag_id, before)
        await refresh_locales_for_content_codes(service, codes, _content_ids(before))
        if codes and tag_id:
            await service._tags().update_one(
                {"_id": before.get("_id")},
                {"$set": {"locale_refresh_pending": False}},
            )
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
