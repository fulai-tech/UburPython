"""量产标签字典批量写入：只写 somni_audio_tag_dictionary 与标签索引。"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from bson import ObjectId
from elasticsearch import NotFoundError
from loguru import logger
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.cache.somni_audio_catalog_cache import clear_somni_audio_catalog_cache
from app.core.bson_util import bson_to_jsonable, utc_now
from app.core.exceptions import AppError
from app.server.somni.audio.index_queue import fence_filter, fence_set, material_index_turn
from app.server.somni.audio.writes import (
    StaleLeaseError,
    _clear_own_delete_tombstone,
    _is_superseded_by_newer_delete,
    _record_delete_tombstone,
    refresh_locales_for_content_codes,
)

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
    # 与更新、删除共用同一把锁，避免创建还在写索引时，更新先成功又被创建时的文档盖掉。
    async with material_index_turn(service, _tag_key(str(tag_oid))) as fence:
        if fence is not None:
            doc["fence_token"] = fence
        try:
            result = await service._tags().insert_one(doc)
        except asyncio.CancelledError:
            _clear_cancellation()
            await _compensate_create_failure(service, str(tag_oid), doc, fence=fence)
            raise
        except DuplicateKeyError as exc:
            raise ValueError("标签已存在：type + code + language") from exc
        except Exception as exc:
            compensated = await _compensate_create_failure(
                service, str(tag_oid), doc, fence=fence
            )
            if not compensated:
                raise TagCreateUncertainError(str(tag_oid), exc) from exc
            raise
        tag_id = str(result.inserted_id)
        saved = {**doc, "_id": result.inserted_id}
        try:
            still_ours = await _index_tag(service, tag_id, saved)
            if not still_ours:
                # 租约已被接管：文档还在就留下新请求的内容，不再清待刷新标记。
                current = await service._tags().find_one({"_id": result.inserted_id})
                if current is None:
                    raise ValueError(f"标签创建未完成：{tag_id}")
                return tag_id
            await _refresh_content_labels(service, saved)
            await _clear_locale_refresh_pending(service, result.inserted_id, saved)
        except asyncio.CancelledError:
            _clear_cancellation()
            await _compensate_create_failure(service, tag_id, saved, fence=fence)
            raise
        except Exception as exc:
            compensated = await _compensate_create_failure(
                service, tag_id, saved, fence=fence
            )
            if not compensated:
                raise TagCreateUncertainError(tag_id, exc) from exc
            raise
        return tag_id


def _tag_key(tag_id: str) -> str:
    """标签写入的串行键与墓碑键；与原料键加前缀隔开。"""
    return f"tag:{tag_id}"


async def _update_one(service: Any, tag_id: str, fields: dict[str, Any]) -> None:
    oid = _parse_id(tag_id)
    payload = _normalize(fields)
    _reject_blank_identity(payload)
    if not payload:
        raise ValueError("没有要更新的字段")
    # 同一标签的 Mongo 写、ES 补丁、补偿按顺序进行，ES 版本号与 Mongo 版本号一致。
    async with material_index_turn(service, _tag_key(tag_id)) as fence:
        await _update_one_locked(service, tag_id, oid, payload, fence)


async def _update_one_locked(
    service: Any,
    tag_id: str,
    oid: ObjectId,
    payload: dict[str, Any],
    fence: int | None,
) -> None:
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
    payload["version"] = int(base_version or 0) + 1
    written_fields = _own_fields(payload)
    # 每笔写入一个唯一操作 ID，并按字段记在 _ops 上。补偿时只撤销仍由本笔占有的字段。
    op = uuid.uuid4().hex
    update_set = {**payload, **fence_set(fence), **_ownership(written_fields, op)}
    saved = before
    try:
        saved = await service._tags().find_one_and_update(
            {"_id": oid, "version": base_version, **fence_filter(fence)},
            {"$set": update_set},
            return_document=ReturnDocument.AFTER,
        )
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_update_failure(
            service, tag_id, before=before, after=saved, op=op, written_fields=written_fields
        )
        raise
    except DuplicateKeyError as exc:
        raise ValueError("标签已存在：type + code + language") from exc
    except Exception:
        await _compensate_update_failure(
            service, tag_id, before=before, after=saved, op=op, written_fields=written_fields
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
            service, tag_id, before=before, after=saved, op=op, written_fields=written_fields
        )
        raise
    except Exception:
        await _compensate_update_failure(
            service, tag_id, before=before, after=saved, op=op, written_fields=written_fields
        )
        raise


async def _delete_one(service: Any, tag_id: str) -> None:
    oid = _parse_id(tag_id)
    async with material_index_turn(service, _tag_key(tag_id)) as fence:
        await _delete_one_locked(service, tag_id, oid, fence)


async def _delete_one_locked(
    service: Any,
    tag_id: str,
    oid: ObjectId,
    fence: int | None,
) -> None:
    before = await service._tags().find_one({"_id": oid})
    key = _tag_key(tag_id)
    if before is None:
        # 标签文档已经没了。先记墓碑，再删掉还留着的索引。索引 id 就是本次请求的 id。
        await _record_delete_tombstone(service, key, fence)
        if await _delete_index(service, tag_id):
            return
        raise ValueError(f"标签不存在：{tag_id}")
    if fence is not None and int(before.get("fence_token") or 0) > fence:
        raise StaleLeaseError(f"标签已被更新的租约接管，写入被拒绝：{tag_id}")
    # 先记墓碑：之后任何旧删除的补偿都能认出"已被更新的删除接管"，不再复活标签。
    await _record_delete_tombstone(service, key, fence)
    try:
        result = await service._tags().delete_one({"_id": oid, **fence_filter(fence)})
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_delete_failure(service, tag_id, before, fence=fence)
        raise
    except Exception:
        await _compensate_delete_failure(service, tag_id, before, fence=fence)
        raise
    if result.deleted_count == 0:
        await _clear_own_delete_tombstone(service, key, fence)
        still = await service._tags().find_one({"_id": oid})
        if still is None:
            raise ValueError(f"标签不存在：{tag_id}")
        raise StaleLeaseError(f"标签已被更新的租约接管，写入被拒绝：{tag_id}")
    try:
        await _delete_index(service, tag_id)
        await _refresh_content_labels(service, before)
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_delete_failure(service, tag_id, before, fence=fence)
        raise
    except Exception:
        await _compensate_delete_failure(service, tag_id, before, fence=fence)
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
    for key in (
        "_id",
        "id",
        "name_en",
        "name_vector",
        "name_en_vector",
        "locale_refresh_pending",
        "_ops",
        "fence_token",
    ):
        payload.pop(key, None)
    parent = payload.pop("parent_id", None)
    payload.pop("parent_tag_id", None)
    payload["parent_id"] = None if parent in (None, "") else str(parent)
    language = str(payload.get("language") or "").strip().lower()
    payload["language"] = language or "zh"
    return payload


async def _tag_version_not_superseded(service: Any, tag_id: str, version: int) -> bool:
    """标签还在，且库里的版本不比本次快照更新，才允许整份写入 ES。"""
    current = await service._tags().find_one({"_id": ObjectId(tag_id)})
    if current is None:
        return False
    return int(current.get("version") or 0) <= version


async def _index_tag(service: Any, tag_id: str, doc: dict[str, Any]) -> bool:
    """整份写入标签索引。返回这次快照是否仍是库里的最新版本。"""
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return True
    version = int(doc.get("version") or 0)
    payload = _tag_source(doc)
    _apply_tag_vector(payload, await _vector_for(service, tag_id, payload))
    # 版本 0 是新建标签的默认版本，也要比较。算向量期间可能已被更新请求推进。
    if not await _tag_version_not_superseded(service, tag_id, version):
        return False
    try:
        await es_search._client.index(
            index=es_search.tag_dictionary_index,
            id=tag_id,
            document=payload,
        )
    except Exception as exc:
        raise RuntimeError(f"标签索引写入失败：{exc}") from exc
    if not await _tag_version_not_superseded(service, tag_id, version):
        await _index_current_tag(service, tag_id)
        return False
    return True


async def _index_current_tag(service: Any, tag_id: str) -> None:
    """整份写入落地时版本已变：标签已删则删索引，否则按当前名字重算向量再整份写入。

    向量算完后要再对一次版本。期间被改掉或删掉就重来，避免旧向量盖住新名字，
    也避免把已经删除的标签重新写回索引。
    """
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return
    for _ in range(4):
        current = await service._tags().find_one({"_id": ObjectId(tag_id)})
        if current is None:
            await _delete_index(service, tag_id)
            return
        version = int(current.get("version") or 0)
        payload = _tag_source(current)
        _apply_tag_vector(payload, await _vector_for(service, tag_id, payload))
        if not await _tag_version_not_superseded(service, tag_id, version):
            continue
        try:
            await es_search._client.index(
                index=es_search.tag_dictionary_index,
                id=tag_id,
                document=payload,
            )
        except Exception as exc:
            raise RuntimeError(f"标签索引写入失败：{exc}") from exc
        if await _tag_version_not_superseded(service, tag_id, version):
            return
    current = await service._tags().find_one({"_id": ObjectId(tag_id)})
    if current is None:
        await _delete_index(service, tag_id)
        return
    raise RuntimeError("标签索引写入失败：版本持续变化")


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


async def _delete_index(service: Any, tag_id: str) -> bool:
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return False
    try:
        await es_search._client.delete(index=es_search.tag_dictionary_index, id=tag_id)
    except NotFoundError:
        return False
    except Exception as exc:
        raise RuntimeError(f"标签索引删除失败：{exc}") from exc
    return True


async def _compensate_create_failure(
    service: Any,
    tag_id: str,
    doc: dict[str, Any],
    *,
    fence: int | None = None,
) -> bool:
    """创建失败时只删除仍属于本次创建的文档。版本或租约编号已被推进则留下新数据。"""
    try:
        version = int(doc.get("version") or 0)
        result = await service._tags().delete_one(
            {"_id": ObjectId(tag_id), "version": version, **fence_filter(fence)}
        )
        if not getattr(result, "deleted_count", 0):
            return True
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
    op: str,
    written_fields: dict[str, Any],
) -> None:
    """更新失败的补偿：只撤销本笔（op）仍占有的字段。

    - 字段的 _ops.<字段> 仍等于本笔操作 ID，说明这个字段最后一次由本笔写入，改回写入前的值。
    - 字段已被别的请求写过（_ops 不再是本笔），或本笔根本没落库，都不动它。
    - 版本号只增不减；索引与检索文本最后以数据库当前记录为准，不用快照。
    """
    try:
        oid = ObjectId(tag_id)
        tags = service._tags()
        codes = _content_codes(before) | _content_codes(after or {})
        for name in written_fields:
            await tags.update_one(
                {"_id": oid, f"_ops.{name}": op},
                {"$set": {name: before.get(name)}, "$unset": {f"_ops.{name}": ""}},
            )
        current = await tags.find_one({"_id": oid})
        if current is not None:
            await _index_tag(service, tag_id, current)
        await refresh_locales_for_content_codes(
            service,
            codes,
            _content_ids(current) | _content_ids(after or {}),
        )
    except Exception as exc:
        logger.error("标签更新补偿失败，tag_id={}，原因：{}", tag_id, exc)


def _ownership(written_fields: dict[str, Any], op: str) -> dict[str, str]:
    """本笔写入对每个业务字段的归属标记，写成 _ops.<字段>。"""
    return {f"_ops.{name}": op for name in written_fields}


async def _compensate_delete_failure(
    service: Any,
    tag_id: str,
    before: dict[str, Any],
    *,
    fence: int | None = None,
) -> None:
    """删除失败：只在标签确实不存在、且没有更新的删除接管时，才用快照插回去。

    插入而不是覆盖：库里已有同 _id 文档（例如别的请求刚编辑过）时直接放弃，不覆写。
    """
    try:
        if await _is_superseded_by_newer_delete(service, _tag_key(tag_id), fence):
            return
        restored = {key: value for key, value in before.items() if key != "_id"}
        restored.update(fence_set(fence))
        codes = _content_codes(before)
        if codes:
            restored["locale_refresh_pending"] = True
        try:
            await service._tags().insert_one({"_id": before.get("_id"), **restored})
        except DuplicateKeyError:
            # 文档仍在：删除没有生效（或已被别人改过），保持现状。
            return
        # 插入与前面的检查之间可能已有更新的删除接管：复检后撤回自己刚插回的文档。
        if await _is_superseded_by_newer_delete(service, _tag_key(tag_id), fence):
            await service._tags().delete_one({"_id": before.get("_id"), **fence_filter(fence)})
            return
        await _index_tag(service, tag_id, before)
        await refresh_locales_for_content_codes(service, codes, _content_ids(before))
        if codes:
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
