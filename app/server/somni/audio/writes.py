"""量产音频原料批量写入。

原料表/原料索引只存状态、来源、标签和 embedding。
标题、描述、短描述、audio_url、cover_url 与检索向量写语言表/语言索引。
"""

from __future__ import annotations

import asyncio
from typing import Any

from bson import ObjectId
from elasticsearch import NotFoundError
from google.protobuf.json_format import MessageToDict
from loguru import logger
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.cache.somni_audio_catalog_cache import clear_somni_audio_catalog_cache
from app.core.bson_util import bson_to_jsonable, utc_now
from app.core.exceptions import AppError
from app.es.somni_docs import (
    build_tag_name_by_code_language,
    extract_sleep_stage_names,
    locale_source_for_es,
    material_content_form_tag_codes,
    material_is_enabled,
    material_tag_labels_from_dictionary,
)
from app.server.somni.audio.index_queue import fence_filter, fence_set, material_index_turn

AUDIO_WRITE_BATCH_LIMIT = 200
_DELETE_TOMBSTONES_COLLECTION = "somni_audio_delete_tombstones"


class _MemoryTombstones:
    """无 _db 的轻量测试替身使用的同构内存墓碑表。"""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        doc = self._docs.get(str(query.get("_id") or ""))
        return dict(doc) if doc is not None else None

    async def replace_one(
        self,
        query: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> Any:
        doc_id = str(query.get("_id") or "")
        existing = self._docs.get(doc_id)
        allowed = self._matches(existing, query) if existing is not None else False
        if existing is not None and not allowed:
            if upsert:
                raise DuplicateKeyError("dup tombstone _id")
            return ReturnDocument.AFTER
        if existing is None and not upsert:
            return None
        self._docs[doc_id] = {"_id": doc_id, **replacement}
        return self._docs[doc_id]

    async def delete_one(self, query: dict[str, Any]) -> Any:
        doc_id = str(query.get("_id") or "")
        existing = self._docs.get(doc_id)
        if existing is not None and existing.get("fence_token") == query.get("fence_token"):
            self._docs.pop(doc_id, None)

    @staticmethod
    def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
        or_clauses = query.get("$or")
        if not or_clauses:
            return True
        token = doc.get("fence_token")
        for clause in or_clauses:
            cond = clause.get("fence_token")
            if cond is None and token is None:
                return True
            if isinstance(cond, dict) and token is not None and token <= cond["$lte"]:
                return True
        return False


_FALLBACK_TOMBSTONE_ATTR = "_fallback_delete_tombstones"


def _delete_tombstones(service: Any) -> Any:
    getter = getattr(service, "_delete_tombstones", None)
    if callable(getter):
        return getter()
    db_getter = getattr(service, "_db", None)
    if callable(db_getter):
        # 启用租约（有 fence）时墓碑库必须可用：失败直接抛出，不静默退回进程内存。
        db = db_getter()
        if isinstance(db, dict):
            if _DELETE_TOMBSTONES_COLLECTION in db:
                return db[_DELETE_TOMBSTONES_COLLECTION]
        elif db is not None:
            return db[_DELETE_TOMBSTONES_COLLECTION]
    # 仅无 _db 的测试替身走这里：挂在服务对象自身上，同一服务多次取到的是同一份。
    store = getattr(service, _FALLBACK_TOMBSTONE_ATTR, None)
    if store is None:
        store = _MemoryTombstones()
        setattr(service, _FALLBACK_TOMBSTONE_ATTR, store)
    return store


async def _record_delete_tombstone(
    service: Any, material_id: str, fence: int | None
) -> None:
    if fence is None:
        return
    tombstones = _delete_tombstones(service)
    try:
        await tombstones.replace_one(
            {"_id": material_id, **fence_filter(fence)},
            {"fence_token": fence, "deleted_at": utc_now()},
            upsert=True,
        )
    except DuplicateKeyError as exc:
        raise _write_rejected(fence, material_id) from exc


async def _is_superseded_by_newer_delete(
    service: Any, material_id: str, fence: int | None
) -> bool:
    if fence is None or not material_id:
        return False
    doc = await _delete_tombstones(service).find_one({"_id": material_id})
    return doc is not None and int(doc.get("fence_token") or 0) > fence


async def _clear_own_delete_tombstone(
    service: Any, material_id: str, fence: int | None
) -> None:
    if fence is None or not material_id:
        return
    await _delete_tombstones(service).delete_one(
        {"_id": material_id, "fence_token": fence}
    )


class AudioCreateUncertainError(RuntimeError):
    """创建失败且补偿未完成；material_id 对调用方仍有核对价值。"""

    def __init__(self, material_id: str, cause: Exception) -> None:
        super().__init__(str(cause) or cause.__class__.__name__)
        self.material_id = material_id


class StaleLeaseError(RuntimeError):
    """本次写入持有的租约编号已落后，原料已被更新的租约接管，写入被拒绝。"""


def _write_rejected(fence: int | None, material_id: str) -> Exception:
    if fence is None:
        return ValueError(f"音频原料不存在：{material_id}")
    return StaleLeaseError(f"原料已被更新的租约接管，写入被拒绝：{material_id}")


_MATERIAL_FIELDS = (
    "status",
    "operation_type",
    "source",
    "vip",
    "created_by",
    "updated_by",
    "origin_file_name",
    "acoustic_metrics",
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
            elif name == "origin_file_name":
                value = str(value or "").strip()
            fields[name] = value
    for name in _TAG_FIELDS:
        if message.HasField(name):
            items = [
                MessageToDict(item, preserving_proto_field_name=True)
                for item in getattr(message, name).items
            ]
            if name == "content_form_tags":
                items = [_content_form_tag_for_storage(item) for item in items]
            fields[name] = items
    if message.HasField("acoustic_metrics"):
        fields["acoustic_metrics"] = _acoustic_metrics_for_storage(message.acoustic_metrics)
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
            if isinstance(exc, AudioCreateUncertainError):
                material_id = exc.material_id
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
    material_oid = ObjectId()
    material["_id"] = material_oid
    try:
        result = await service._materials().insert_one(material)
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_create_failure(service, str(material_oid), None)
        raise
    except Exception as exc:
        compensated = await _compensate_create_failure(
            service,
            str(material_oid),
            None,
        )
        if not compensated:
            raise AudioCreateUncertainError(str(material_oid), exc) from exc
        raise
    material_id = str(result.inserted_id)
    saved = {**material, "_id": result.inserted_id}
    locale: dict[str, Any] | None = None
    completed = False
    fence_held: int | None = None
    try:
        async with material_index_turn(service, material_id) as fence:
            fence_held = fence
            if fence is not None:
                saved = (
                    await service._materials().find_one_and_update(
                        {"_id": result.inserted_id, **fence_filter(fence)},
                        {"$set": fence_set(fence)},
                        return_document=ReturnDocument.AFTER,
                    )
                    or {**saved, **fence_set(fence)}
                )
            locale = await _save_locale_and_sync_create(
                service, material_id, saved, fields, audio_name, language, fence=fence
            )
        completed = True
    except asyncio.CancelledError:
        if not completed:
            _clear_cancellation()
            await _compensate_create_failure(
                service, material_id, locale, fence=fence_held
            )
        raise
    except Exception as exc:
        if not completed:
            compensated = await _compensate_create_failure(
                service, material_id, locale, fence=fence_held
            )
            if not compensated:
                raise AudioCreateUncertainError(material_id, exc) from exc
        raise
    return material_id


async def _save_locale_and_sync_create(
    service: Any,
    material_id: str,
    saved: dict[str, Any],
    fields: dict[str, Any],
    audio_name: str,
    language: str,
    *,
    fence: int | None = None,
) -> dict[str, Any]:
    locale = await _save_locale(
        service,
        material_id,
        {
            **_CREATE_LOCALE_DEFAULTS,
            **_pick(fields, _LOCALE_FIELDS),
            "audio_name": audio_name,
            "language": language,
        },
        fence=fence,
    )
    await _sync_after_write(
        service,
        material_id,
        saved,
        changed=set(fields) | set(_CREATE_LOCALE_DEFAULTS) | {"audio_name"},
        wrote_locale=locale,
        recompute_locale_vector=True,
        create=True,
        fence=fence,
    )
    return locale


async def _update_one(service: Any, material_id: str, fields: dict[str, Any]) -> None:
    _parse_id(material_id)
    material_fields = _pick(fields, _MATERIAL_FIELDS, _TAG_FIELDS, ("embedding",))
    acoustic_metrics = material_fields.pop("acoustic_metrics", None)
    if isinstance(acoustic_metrics, dict):
        material_fields.update(
            {
                f"acoustic_metrics.{name}": value
                for name, value in acoustic_metrics.items()
            }
        )
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
    async with material_index_turn(service, material_id) as fence:
        await _update_one_locked(
            service, material_id, fields, material_fields, locale_fields, fence
        )


async def _update_one_locked(
    service: Any,
    material_id: str,
    fields: dict[str, Any],
    material_fields: dict[str, Any],
    locale_fields: dict[str, Any],
    fence: int | None = None,
) -> None:
    oid = _parse_id(material_id)
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
            update: dict[str, Any] = {"$set": {**material_fields, **fence_set(fence)}}
            if clear_embedding:
                update["$unset"] = {"embedding": ""}
            material_written = True
            saved = await service._materials().find_one_and_update(
                {"_id": oid, **fence_filter(fence)},
                update,
                return_document=ReturnDocument.AFTER,
            )
        elif fence is not None:
            material_written = True
            saved = await service._materials().find_one_and_update(
                {"_id": oid, **fence_filter(fence)},
                {"$set": fence_set(fence)},
                return_document=ReturnDocument.AFTER,
            )
        if saved is None:
            raise _write_rejected(fence, material_id)
        if locale_fields:
            locale_written = True
            locale = await _save_locale(service, material_id, locale_fields, fence=fence)
        await _sync_after_write(
            service,
            material_id,
            saved,
            changed=set(fields) | ({"updated_at"} if touch_material else set()),
            wrote_locale=locale,
            recompute_locale_vector=_recompute_locale_vector(fields),
            touch_material=touch_material,
            fence=fence,
        )
    except asyncio.CancelledError:
        if material_written or locale_written:
            _clear_cancellation()
            await _compensate_update_failure(
                service,
                material_id,
                before_material=before_material,
                before_locale=before_locale,
                wrote_locale=locale,
                locale_language=str(locale_fields.get("language") or ""),
                touched_material=material_written,
                touched_locale=locale_written,
                fence=fence,
            )
        raise
    except Exception:
        if material_written or locale_written:
            await _compensate_update_failure(
                service,
                material_id,
                before_material=before_material,
                before_locale=before_locale,
                wrote_locale=locale,
                locale_language=str(locale_fields.get("language") or ""),
                touched_material=material_written,
                touched_locale=locale_written,
                fence=fence,
            )
        raise


async def _delete_one(service: Any, material_id: str) -> None:
    oid = _parse_id(material_id)
    async with material_index_turn(service, material_id) as fence:
        await _delete_one_locked(service, material_id, oid, fence)


async def _delete_one_locked(
    service: Any,
    material_id: str,
    oid: ObjectId,
    fence: int | None = None,
) -> None:
    locales = _locale_collection(service)
    before_material = await service._materials().find_one({"_id": oid})
    if before_material is None:
        # 已被前序请求删除：仍记下本次删除的墓碑，阻止更旧删除的补偿把它复活。
        await _record_delete_tombstone(service, material_id, fence)
        raise ValueError(f"音频原料不存在：{material_id}")
    if fence is not None and int(before_material.get("fence_token") or 0) > fence:
        raise _write_rejected(fence, material_id)
    rows = await locales.find({"material_id": material_id}).to_list(length=None)
    await _record_delete_tombstone(service, material_id, fence)
    try:
        result = await service._materials().delete_one({"_id": oid, **fence_filter(fence)})
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_delete_failure(
            service,
            before_material,
            rows,
            locales_deleted=False,
            fence=fence,
        )
        raise
    except Exception:
        await _compensate_delete_failure(
            service,
            before_material,
            rows,
            locales_deleted=False,
            fence=fence,
        )
        raise
    if result.deleted_count == 0:
        await _clear_own_delete_tombstone(service, material_id, fence)
        raise _write_rejected(fence, material_id)
    locales_deleted = False
    try:
        locales_deleted = True
        await locales.delete_many({"material_id": material_id, **fence_filter(fence)})
        await _delete_index(service, material_id)
        for row in rows:
            await _delete_locale_index(service, str(row.get("_id") or ""))
    except asyncio.CancelledError:
        _clear_cancellation()
        await _compensate_delete_failure(
            service,
            before_material,
            rows,
            locales_deleted=locales_deleted,
            fence=fence,
        )
        raise
    except Exception:
        await _compensate_delete_failure(
            service,
            before_material,
            rows,
            locales_deleted=locales_deleted,
            fence=fence,
        )
        raise


_ACOUSTIC_NUMBER_FIELDS = (
    "duration_seconds",
    "transient_ratio",
    "transient_max_delta_db",
    "voice_presence_ratio",
)


def _acoustic_metrics_for_storage(metrics: Any) -> dict[str, Any]:
    """AcousticMetrics → 库里的对象。tempo_no_beat 写回 tempo_bpm: null。"""
    out: dict[str, Any] = {}
    for name in _ACOUSTIC_NUMBER_FIELDS:
        if metrics.HasField(name):
            out[name] = float(getattr(metrics, name))
    if metrics.HasField("tempo_bpm"):
        out["tempo_bpm"] = float(metrics.tempo_bpm)
    elif metrics.HasField("tempo_no_beat") and metrics.tempo_no_beat:
        out["tempo_bpm"] = None
    return out


def _content_form_tag_for_storage(item: dict[str, Any]) -> dict[str, Any]:
    """proto 字段是 parent_tag_id，库和后台表单用 parent_id。"""
    row = dict(item)
    parent = str(row.pop("parent_tag_id", "") or "").strip()
    row.pop("parent_id", None)
    if parent:
        row["parent_id"] = parent
    return row


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


def _clear_cancellation() -> None:
    """取消会粘在当前任务上，补偿里的 await 会立刻再被取消。先清掉再回滚。"""
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()


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


def _create_still_owned(fence_token: Any, fence: int | None) -> bool:
    """没有更新的租约编号时，这份创建写入仍属于本次补偿。"""
    if fence is None:
        return fence_token is None
    return fence_token is None or int(fence_token) <= fence


async def _compensate_create_failure(
    service: Any,
    material_id: str,
    locale: dict[str, Any] | None,
    *,
    fence: int | None = None,
) -> bool:
    """创建失败时只删除仍属于本次创建的原料。编号已被推进则留下新数据。"""
    try:
        oid = ObjectId(material_id)
        if fence is None:
            # 还没盖上编号：只删未被后续请求接管的文档。
            material_query: dict[str, Any] = {"_id": oid, "fence_token": None}
            locale_query: dict[str, Any] = {
                "material_id": material_id,
                "fence_token": None,
            }
        else:
            material_query = {"_id": oid, **fence_filter(fence)}
            locale_query = {"material_id": material_id, **fence_filter(fence)}
        result = await service._materials().delete_one(material_query)
        if not getattr(result, "deleted_count", 0):
            return True
        locales = _locale_collection(service)
        locale_rows = await locales.find(locale_query).to_list(length=None)
        if locale is not None and _create_still_owned(locale.get("fence_token"), fence):
            if not any(row.get("_id") == locale.get("_id") for row in locale_rows):
                locale_rows.append(locale)
        await locales.delete_many(locale_query)
        await _delete_index(service, material_id)
        for row in locale_rows:
            locale_id = str(row.get("_id") or "")
            if locale_id:
                await _delete_locale_index(service, locale_id)
        return True
    except Exception as exc:
        logger.error(
            "创建补偿失败，material_id={}，原因：{}",
            material_id,
            exc,
        )
        return False


def _positive_count(result: Any, attr: str) -> bool:
    count = getattr(result, attr, 1)
    if isinstance(count, int):
        return count > 0
    return True


async def _compensate_update_failure(
    service: Any,
    material_id: str,
    *,
    before_material: dict[str, Any],
    before_locale: dict[str, Any] | None,
    wrote_locale: dict[str, Any] | None,
    locale_language: str,
    touched_material: bool,
    touched_locale: bool,
    fence: int | None = None,
) -> None:
    """更新失败：只撤销本租约写入的原料与语言行；之后按数据库当前状态重刷 ES，不用快照。"""
    try:
        oid = _parse_id(material_id)
        owned_filter = {"fence_token": fence} if fence is not None else {}
        owned = True
        locale_owned = True
        if touched_material:
            owned = await _restore_material_if_owned(service, oid, before_material, fence)
        if touched_locale:
            locales = _locale_collection(service)
            if before_locale is None:
                target_query = {
                    "material_id": material_id,
                    "language": locale_language,
                    **owned_filter,
                }
                current_loc = await locales.find_one(target_query)
                del_res = await locales.delete_many(target_query)
                locale_owned = _positive_count(del_res, "deleted_count")
                if locale_owned:
                    locale_id = (current_loc or wrote_locale or {}).get("_id")
                    if locale_id is not None:
                        await _delete_locale_index(service, str(locale_id))
            elif before_locale is not None:
                locale_oid = before_locale["_id"]
                restore_locale = {
                    key: value
                    for key, value in before_locale.items()
                    if key not in ("_id", "fence_token")
                }
                restore_locale.update(fence_set(fence))
                loc_res = await locales.replace_one(
                    {"_id": locale_oid, **owned_filter},
                    restore_locale,
                )
                locale_owned = _positive_count(loc_res, "matched_count")
        # 原料文档不依赖标签词典，先恢复；语言索引要读词典，放后面，
        # 词典不可用时才不会拦住原料文档的恢复。
        # 索引一律以数据库当前内容为准：本租约已不是最后写入者时，索引留给新持有者。
        current = await service._materials().find_one({"_id": oid})
        if current is not None:
            not_superseded = (
                fence is None or (current.get("fence_token") or 0) <= fence
            )
            if touched_material and owned and not_superseded:
                await _index_material_document(
                    service, material_id, _material_index_body(current)
                )
            # 语言行若仍是本租约回滚后的版本（locale_owned），它的索引就该跟着刷新，
            # 不能因为原料已被更新的租约接管而跳过（否则 ES 停留在已回滚的脏标题）。
            if (touched_locale and locale_owned) or (
                not_superseded and touched_material and owned
            ):
                await _reindex_material_locales(service, material_id, current)
    except Exception as exc:
        logger.error(
            "更新补偿失败，material_id={}，原因：{}",
            material_id,
            exc,
        )


async def _restore_material_if_owned(
    service: Any,
    oid: ObjectId,
    before_material: dict[str, Any],
    fence: int | None,
) -> bool:
    """库里仍是本租约写入的那一版才回滚。返回是否真的回滚了。

    回滚后编号仍保持本租约的值，不能用快照里的旧编号覆盖，否则编号会倒退。
    """
    restore = {
        key: value for key, value in before_material.items() if key not in ("_id", "fence_token")
    }
    restore.update(fence_set(fence))
    owned_filter = {"fence_token": fence} if fence is not None else {}
    result = await service._materials().replace_one({"_id": oid, **owned_filter}, restore)
    return bool(getattr(result, "matched_count", 1))


async def _compensate_delete_failure(
    service: Any,
    before_material: dict[str, Any],
    locale_rows: list[dict[str, Any]],
    *,
    locales_deleted: bool,
    fence: int | None = None,
) -> None:
    """删除失败：幂等恢复原料与语言行，并尽量重刷 ES。

    原料恢复带编号门禁与删除标记校验：若已有更高编号的写入或删除，不复活、不覆盖新数据。
    """
    material_id = str(before_material.get("_id") or "")
    try:
        if await _is_superseded_by_newer_delete(service, material_id, fence):
            return
        material_oid = before_material["_id"]
        restore_material = {
            key: value
            for key, value in before_material.items()
            if key not in ("_id", "fence_token")
        }
        restore_material.update(fence_set(fence))
        await service._materials().replace_one(
            {"_id": material_oid, **fence_filter(fence)},
            restore_material,
            upsert=True,
        )
        if await _is_superseded_by_newer_delete(service, material_id, fence):
            await service._materials().delete_one(
                {"_id": material_oid, **fence_filter(fence)}
            )
            return
        await _clear_own_delete_tombstone(service, material_id, fence)
        if locales_deleted and locale_rows:
            locales = _locale_collection(service)
            for row in locale_rows:
                locale_oid = row.get("_id")
                if locale_oid is None:
                    continue
                restore_locale = {
                    key: value
                    for key, value in row.items()
                    if key not in ("_id", "fence_token")
                }
                restore_locale.update(fence_set(fence))
                await locales.replace_one(
                    {"_id": locale_oid, **fence_filter(fence)},
                    restore_locale,
                    upsert=True,
                )
        if material_id:
            # 与刚恢复到库里的文档一致：编号是恢复时写入的本租约编号，不是快照里的旧编号。
            await _index_material_document(
                service,
                material_id,
                _material_index_body({**before_material, **fence_set(fence)}),
            )
        names = (
            await _content_form_names(service, before_material) if locale_rows else {}
        )
        for row in locale_rows:
            locale_id = str(row.get("_id") or "")
            source = await _locale_source_for_index(
                service,
                {**row, **fence_set(fence)},
                material=before_material,
                names=names,
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
    *,
    fence: int | None = None,
) -> dict[str, Any]:
    """按 material_id + language 写入。第一次选定语言时，把空语言行改成该语言。"""
    collection = _locale_collection(service)
    language = str(fields.get("language") or "").strip().lower()
    now = utc_now()
    payload = {key: value for key, value in fields.items() if key != "language"}
    payload["language"] = language
    payload["update_time"] = now
    payload.update(fence_set(fence))
    if language:
        current = await collection.find_one({"material_id": material_id, "language": language})
        if current is not None and fence is not None:
            saved = await collection.find_one_and_update(
                {"_id": current["_id"], **fence_filter(fence)},
                {"$set": payload},
                return_document=ReturnDocument.AFTER,
            )
            if saved is None:
                raise _write_rejected(fence, material_id)
            saved["material_id"] = material_id
            return saved
        if current is None:
            blank = await collection.find_one({"material_id": material_id, "language": ""})
            if blank is not None:
                saved = await collection.find_one_and_update(
                    {"_id": blank["_id"], **fence_filter(fence)},
                    {"$set": payload},
                    return_document=ReturnDocument.AFTER,
                )
                if saved is None and fence is not None:
                    raise _write_rejected(fence, material_id)
                if saved is not None:
                    saved["material_id"] = material_id
                    return saved
    try:
        saved = await collection.find_one_and_update(
            {"material_id": material_id, "language": language, **fence_filter(fence)},
            {
                "$set": payload,
                "$setOnInsert": {"material_id": material_id, "create_time": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError as exc:
        raise _write_rejected(fence, material_id) from exc
    if saved is not None:
        saved["material_id"] = material_id
        if fence is not None and saved.get("_id") is not None:
            siblings = await collection.find(
                {"material_id": material_id, "language": language}
            ).to_list(length=None)
            if any((row.get("fence_token") or 0) > fence for row in siblings):
                await collection.delete_one({"_id": saved["_id"], "fence_token": fence})
                raise _write_rejected(fence, material_id)
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
    fence: int | None = None,
) -> None:
    es_search = getattr(service, "_es_search", None)
    if es_search is None:
        return
    if create or touch_material or any(name in changed for name in _TAG_FIELDS):
        body = _material_index_body(material)
        if create:
            await _index_material_document(service, material_id, body)
        else:
            await _patch_material_index(
                service, material_id, body, changed=changed, fence=fence
            )
    tags_changed = "content_form_tags" in changed
    index_locales = bool(
        wrote_locale is not None and str(wrote_locale.get("language") or "").strip()
    ) or (tags_changed and not create)
    names = await _content_form_names(service, material) if index_locales else {}
    if wrote_locale is not None and str(wrote_locale.get("language") or "").strip():
        if create:
            source = await _locale_source_for_index(
                service, wrote_locale, material=material, names=names
            )
            if source is not None:
                await _index_locale_document(service, str(wrote_locale.get("_id") or ""), source)
        else:
            await _patch_locale_index(
                service,
                wrote_locale,
                changed=changed,
                recompute_vector=recompute_locale_vector or tags_changed,
                material=material,
                names=names,
                fence=fence,
            )
    if tags_changed and not create:
        await _refresh_other_locale_texts(
            service,
            material_id,
            material,
            skip_locale_id=str((wrote_locale or {}).get("_id") or ""),
            names=names,
            fence=fence,
        )
    if "status" in changed:
        await _patch_locales_material_status(
            service, material_id, material_is_enabled(material), fence=fence
        )


async def _patch_material_index(
    service: Any,
    material_id: str,
    body: dict[str, Any],
    *,
    changed: set[str],
    fence: int | None = None,
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
            guard=_es_guard("fence_token", fence),
        )
    except Exception as exc:
        raise RuntimeError(f"检索索引写入失败：{exc}") from exc
    if not updated:
        await _index_material_document(service, material_id, body)


def _es_guard(field: str, value: int | None) -> tuple[str, int] | None:
    """ES 文档上记录的编号比本次大时，局部更新直接 noop，防止旧写入盖掉新写入。"""
    return None if value is None else (field, value)


def _material_index_body(material: dict[str, Any]) -> dict[str, Any]:
    flat = bson_to_jsonable(material)
    for key in list(flat):
        if key in _ES_SKIP:
            flat.pop(key, None)
    flat["sleep_stage_names"] = extract_sleep_stage_names(flat)
    return flat


async def _reindex_material_locales(
    service: Any, material_id: str, material: dict[str, Any]
) -> None:
    """按当前库里的语言行，用给定原料快照重写全部语言索引。"""
    es_search = getattr(service, "_es_search", None)
    if es_search is None or not getattr(es_search, "material_locales_index", ""):
        return
    names = await _content_form_names(service, material)
    rows = await _locale_collection(service).find({"material_id": material_id}).to_list(
        length=None
    )
    for row in rows:
        locale_id = str(row.get("_id") or "")
        if not str(row.get("language") or "").strip():
            if locale_id:
                await _delete_locale_index(service, locale_id)
            continue
        if not locale_id:
            continue
        source = await _locale_source_for_index(
            service, row, material=material, names=names
        )
        if source is None:
            await _delete_locale_index(service, locale_id)
        else:
            await _index_locale_document(service, locale_id, source)


async def _fence_not_superseded(collection: Any, doc_id: str, fence: int) -> bool:
    """文档还在，且库里的编号不比本次快照更新，才允许整份写入 ES。

    编号更小或还没有编号的旧行仍要补写索引。编号更大说明已被新请求接管。
    """
    current = await collection.find_one({"_id": _parse_id(doc_id)})
    if current is None:
        return False
    return int(current.get("fence_token") or 0) <= fence


async def _index_material_document(
    service: Any,
    material_id: str,
    body: dict[str, Any],
) -> None:
    fence = body.get("fence_token")
    materials = service._materials()
    if fence is not None and not await _fence_not_superseded(materials, material_id, int(fence)):
        return
    try:
        await service._es_search._client.index(
            index=service._es_search.audio_index,
            id=material_id,
            document=body,
        )
    except Exception as exc:
        raise RuntimeError(f"检索索引写入失败：{exc}") from exc
    if fence is None or await _fence_not_superseded(materials, material_id, int(fence)):
        return
    # 写入落地时编号已被推进：ES 上现在是旧快照，改成库里的当前文档。
    current = await materials.find_one({"_id": _parse_id(material_id)})
    if current is None:
        await _delete_index(service, material_id)
        return
    try:
        await service._es_search._client.index(
            index=service._es_search.audio_index,
            id=material_id,
            document=_material_index_body(current),
        )
    except Exception as exc:
        raise RuntimeError(f"检索索引写入失败：{exc}") from exc


async def refresh_locales_for_content_codes(
    service: Any,
    codes: set[str],
    tag_ids: set[str] | None = None,
) -> None:
    """内容形态标签的名字或启停变了，重算引用这些 code 的语言检索文本。"""
    wanted = {str(code).strip() for code in codes if str(code).strip()}
    if not wanted:
        return
    materials = getattr(service, "_materials", None)
    if not callable(materials):
        return
    wanted_ids = {
        str(tag_id).strip()
        for tag_id in (tag_ids or set())
        if str(tag_id).strip()
    }
    tags = getattr(service, "_tags", None)
    if callable(tags):
        rows = [
            doc
            async for doc in tags().find(
                {"type": "content_form", "code": {"$in": sorted(wanted)}},
                projection={"_id": 1},
            )
        ]
        wanted_ids.update(
            str(doc.get("_id") or "").strip()
            for doc in rows
            if str(doc.get("_id") or "").strip()
        )
    references: list[dict[str, Any]] = [
        {"content_form_tags.code": {"$in": sorted(wanted)}},
    ]
    if wanted_ids:
        references.append({"content_form_tags.tag_id": {"$in": sorted(wanted_ids)}})
    rows = await materials().find(
        {"$or": references},
        {"_id": 1},
    ).to_list(length=None)
    for row in rows:
        material_id = str(row.get("_id") or "")
        if not material_id:
            continue
        async with material_index_turn(service, material_id) as fence:
            # 列表是进锁前的快照。锁内重读，避免盖掉刚改过的标签和启停。
            material = await materials().find_one({"_id": row["_id"]})
            if material is None:
                continue
            if fence is not None and (material.get("fence_token") or 0) > fence:
                continue
            if not (
                wanted.intersection(material_content_form_tag_codes(material))
                or wanted_ids.intersection(_content_form_tag_ids(material))
            ):
                continue
            names = await _content_form_names(service, material)
            await _refresh_other_locale_texts(
                service,
                material_id,
                material,
                skip_locale_id="",
                names=names,
                fence=fence,
            )


async def _content_form_names(
    service: Any,
    material: dict[str, Any] | None,
) -> dict[tuple[str, str], str]:
    """一次查出这些内容形态 code 的全部语言展示名。词典读失败时不写空文本。"""
    codes = material_content_form_tag_codes(material)
    tag_ids = _content_form_tag_ids(material)
    if not codes and not tag_ids:
        return {}
    tags = getattr(service, "_tags", None)
    if not callable(tags):
        raise RuntimeError("标签词典不可用")
    selectors: list[dict[str, Any]] = []
    if codes:
        selectors.append({"code": {"$in": codes}})
    object_ids = [ObjectId(tag_id) for tag_id in tag_ids if ObjectId.is_valid(tag_id)]
    if object_ids:
        selectors.append({"_id": {"$in": object_ids}})
    if not selectors:
        return {}
    selector = selectors[0] if len(selectors) == 1 else {"$or": selectors}
    try:
        rows = [
            doc
            async for doc in tags().find(
                {
                    "type": "content_form",
                    "status": "启用",
                    **selector,
                },
                projection={
                    "code": 1,
                    "name": 1,
                    "language": 1,
                    "type": 1,
                    "status": 1,
                },
            )
        ]
    except Exception as exc:
        raise RuntimeError("标签词典不可用") from exc
    return build_tag_name_by_code_language(bson_to_jsonable(rows))


def _content_form_tag_ids(material: dict[str, Any] | None) -> set[str]:
    items = [] if not material else material.get("content_form_tags") or []
    if not isinstance(items, list):
        return set()
    return {
        tag_id
        for item in items
        if isinstance(item, dict)
        if (tag_id := str(item.get("tag_id") or "").strip())
    }


async def _locale_source_for_index(
    service: Any,
    locale: dict[str, Any],
    *,
    material: dict[str, Any] | None = None,
    names: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any] | None:
    flat = bson_to_jsonable(locale)
    language = str(flat.get("language") or "")
    if names is None:
        names = await _content_form_names(service, material)
    labels = material_tag_labels_from_dictionary(
        material,
        language=language,
        name_by_code_lang=names,
    )
    source = locale_source_for_es(
        flat,
        material_status=None if material is None else material_is_enabled(material),
        tag_labels=labels,
    )
    if source is not None and flat.get("fence_token") is not None:
        source["fence_token"] = int(flat["fence_token"])
    return source


async def _refresh_other_locale_texts(
    service: Any,
    material_id: str,
    material: dict[str, Any],
    *,
    skip_locale_id: str,
    names: dict[tuple[str, str], str],
    fence: int | None = None,
) -> None:
    """内容形态标签变了，其它语言行的检索文本也要重算。"""
    rows = await _locale_collection(service).find({"material_id": material_id}).to_list(
        length=None
    )
    for row in rows:
        if not str(row.get("language") or "").strip():
            continue
        if skip_locale_id and str(row.get("_id") or "") == skip_locale_id:
            continue
        await _patch_locale_index(
            service,
            row,
            changed={"description_text"},
            recompute_vector=True,
            material=material,
            names=names,
            fence=fence,
        )


async def _patch_locale_index(
    service: Any,
    locale: dict[str, Any],
    *,
    changed: set[str],
    recompute_vector: bool,
    material: dict[str, Any] | None = None,
    names: dict[tuple[str, str], str] | None = None,
    fence: int | None = None,
) -> None:
    es_search = service._es_search
    if not getattr(es_search, "material_locales_index", ""):
        return
    locale_id = str(locale.get("_id") or "")
    if not locale_id:
        return
    source = await _locale_source_for_index(
        service, locale, material=material, names=names
    )
    if source is None:
        return
    if fence is not None:
        source["fence_token"] = fence
    fields = {key: source[key] for key in changed if key in source}
    if "material_status" in source:
        fields["material_status"] = source["material_status"]
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
        return
    try:
        updated = await _patch_document(
            es_search._client,
            index=es_search.material_locales_index,
            doc_id=locale_id,
            fields=fields,
            remove=remove,
            guard=_es_guard("fence_token", fence),
        )
    except Exception as exc:
        raise RuntimeError(f"语言索引写入失败：{exc}") from exc
    if not updated:
        await _index_locale_document(service, locale_id, source)


async def _patch_locales_material_status(
    service: Any,
    material_id: str,
    enabled: bool,
    *,
    fence: int | None = None,
) -> None:
    """启停只改各语言文档的 material_status，不重算检索文本和向量。"""
    es_search = getattr(service, "_es_search", None)
    if es_search is None or not getattr(es_search, "material_locales_index", ""):
        return
    rows = await _locale_collection(service).find(
        {"material_id": material_id}, {"_id": 1}
    ).to_list(length=None)
    for row in rows:
        locale_id = str(row.get("_id") or "")
        if not locale_id:
            continue
        try:
            await _patch_document(
                es_search._client,
                index=es_search.material_locales_index,
                doc_id=locale_id,
                fields={"material_status": enabled},
                remove=[],
                guard=_es_guard("fence_token", fence),
            )
        except Exception as exc:
            raise RuntimeError(f"语言索引写入失败：{exc}") from exc


def _locale_vector_field(language: str) -> str:
    if language == "zh":
        return "description_vector"
    if language == "en":
        return "description_vector_en"
    return ""


async def _attach_locale_vector(service: Any, locale_id: str, source: dict[str, Any]) -> None:
    language = str(source.get("language") or "")
    text = str(source.get("description_text") or "").strip()
    source.pop("description_vector", None)
    source.pop("description_vector_en", None)
    vector_field = _locale_vector_field(language)
    if not vector_field or not text:
        return
    vector = await _encode_vector(
        service, locale_id, text, required=False, language=language
    )
    if vector is not None:
        source[vector_field] = vector


async def _material_of_locale(service: Any, row: dict[str, Any]) -> dict[str, Any] | None:
    material_id = str(row.get("material_id") or "")
    materials = getattr(service, "_materials", None)
    if not callable(materials) or not ObjectId.is_valid(material_id):
        return None
    return await materials().find_one({"_id": ObjectId(material_id)})


async def _index_locale_document(service: Any, locale_id: str, source: dict[str, Any]) -> None:
    await _attach_locale_vector(service, locale_id, source)
    fence = source.get("fence_token")
    locales = _locale_collection(service)
    if fence is not None and not await _fence_not_superseded(locales, locale_id, int(fence)):
        return
    try:
        await service._es_search._client.index(
            index=service._es_search.material_locales_index,
            id=locale_id,
            document=source,
        )
    except Exception as exc:
        raise RuntimeError(f"语言索引写入失败：{exc}") from exc
    if fence is None or await _fence_not_superseded(locales, locale_id, int(fence)):
        return
    await _repair_locale_index(service, locale_id)


async def _repair_locale_index(service: Any, locale_id: str) -> None:
    """旧快照盖住语言索引后，按当前语言行和原料重写启停、标签名和向量。

    算向量期间编号又被推进，就丢掉这份快照再读当前行。原料或语言行已经没了，就删索引。
    """
    locales = _locale_collection(service)
    es_search = service._es_search
    for _ in range(4):
        row = await locales.find_one({"_id": _parse_id(locale_id)})
        if row is None:
            await _delete_locale_index(service, locale_id)
            return
        material = await _material_of_locale(service, row)
        if material is None:
            await _delete_locale_index(service, locale_id)
            return
        source = await _locale_source_for_index(service, row, material=material)
        if source is None:
            await _delete_locale_index(service, locale_id)
            return
        await _attach_locale_vector(service, locale_id, source)
        fence = source.get("fence_token")
        if fence is not None and not await _fence_not_superseded(locales, locale_id, int(fence)):
            continue
        try:
            await es_search._client.index(
                index=es_search.material_locales_index,
                id=locale_id,
                document=source,
            )
        except Exception as exc:
            raise RuntimeError(f"语言索引写入失败：{exc}") from exc
        if fence is None or await _fence_not_superseded(locales, locale_id, int(fence)):
            return
    raise RuntimeError("语言索引写入失败：租约编号持续变化")


async def _patch_document(
    client: Any,
    *,
    index: str,
    doc_id: str,
    fields: dict[str, Any],
    remove: list[str],
    guard: tuple[str, int] | None = None,
) -> bool:
    """局部更新。文档不存在时返回 False，调用方再整份写入。

    guard=(字段名, 编号)：文档上该字段已大于编号时整次更新 noop，并把字段写成编号。
    """
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
