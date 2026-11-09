"""把查到的文档收成契约里的 table_data。"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from app.server.somni.ask.errors import AskContextError
from app.server.somni.ask.query_builder import document_roots, readable_roots

_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def parse_filter_key(table_name: str, ref: dict[str, Any]) -> list[str] | None:
    """省略或空数组表示返回全部可读字段。非法路径整次拒绝。"""
    if "filter_key" not in ref:
        return None
    value = ref["filter_key"]
    if value is None or value == "":
        raise AskContextError("INVALID_ARGUMENT", "filter_key 不能为空字符串或 null")
    if value == []:
        return None
    if not isinstance(value, list):
        raise AskContextError("INVALID_ARGUMENT", "filter_key 必须是字符串数组")
    readable = readable_roots(table_name)
    documents = document_roots(table_name)
    paths: list[str] = []
    for item in value:
        if not isinstance(item, str) or _PATH_RE.fullmatch(item) is None or len(item) > 256:
            raise AskContextError("INVALID_ARGUMENT", "filter_key 含有非法路径")
        root, dot, _rest = item.partition(".")
        if root not in readable:
            raise AskContextError("INVALID_ARGUMENT", f"不支持读取字段 {root}")
        if dot and root not in documents:
            raise AskContextError("INVALID_ARGUMENT", f"不支持读取字段 {item}")
        paths.append(item)
    return _uncovered(paths)


def shape_table(
    groups: list[tuple[list[dict[str, Any]], list[str] | None]],
    readable: frozenset[str],
) -> dict[str, Any] | list[Any]:
    """同一张表按 _id 去重。一条记录返回对象，多条返回数组，零条返回空数组。"""
    order: list[str] = []
    slots: dict[str, dict[str, Any]] = {}
    for docs, paths in groups:
        for doc in docs:
            identity = _doc_id(doc)
            slot = slots.get(identity)
            if slot is None:
                order.append(identity)
                slot = {"doc": doc, "paths": [], "full": False}
                slots[identity] = slot
            if paths is None:
                slot["full"] = True
            else:
                slot["paths"].extend(paths)
    records = []
    for identity in order:
        slot = slots[identity]
        paths = None if slot["full"] else _uncovered(slot["paths"])
        records.append(project_document(slot["doc"], paths, readable))
    if len(records) == 1:
        return records[0]
    return records


def project_document(
    doc: dict[str, Any],
    paths: list[str] | None,
    readable: frozenset[str],
) -> dict[str, Any]:
    normalized = _normalize(doc)
    if not isinstance(normalized, dict):
        raise AskContextError("INTERNAL", "记录无法编码")
    if paths is None:
        return {key: normalized[key] for key in normalized if key in readable}
    result: dict[str, Any] = {}
    for path in paths:
        _place(result, normalized, path.split("."), path)
    return result


def _uncovered(paths: list[str]) -> list[str]:
    unique = list(dict.fromkeys(paths))
    return [path for path in unique if not any(path.startswith(other + ".") for other in unique)]


def _place(
    result: dict[str, Any],
    source: dict[str, Any],
    segments: list[str],
    full_path: str,
) -> None:
    key = segments[0]
    rest = segments[1:]
    if key not in source:
        raise AskContextError("NOT_FOUND", f"记录缺少字段 {full_path}")
    value = source[key]
    if not rest:
        result[key] = value
        return
    if isinstance(value, list):
        bucket = result.get(key)
        if not isinstance(bucket, list):
            bucket = [{} for _ in value]
            result[key] = bucket
        if len(bucket) != len(value):
            raise AskContextError("NOT_FOUND", f"记录缺少字段 {full_path}")
        for child, item in zip(bucket, value, strict=True):
            if not isinstance(item, dict) or not isinstance(child, dict):
                raise AskContextError("NOT_FOUND", f"记录缺少字段 {full_path}")
            _place(child, item, rest, full_path)
        return
    if isinstance(value, dict):
        child = result.get(key)
        if not isinstance(child, dict):
            child = {}
            result[key] = child
        _place(child, value, rest, full_path)
        return
    raise AskContextError("NOT_FOUND", f"记录缺少字段 {full_path}")


def _doc_id(doc: dict[str, Any]) -> str:
    if not isinstance(doc, dict):
        raise AskContextError("INTERNAL", "记录缺少 _id")
    raw = doc.get("_id")
    if isinstance(raw, ObjectId):
        return str(raw)
    if isinstance(raw, str) and raw:
        return raw
    raise AskContextError("INTERNAL", "记录缺少 _id")


def _normalize(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items() if key != "public_url"}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if type(value) is bool:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AskContextError("INTERNAL", "结果含有无法编码的数值")
        return value
    if isinstance(value, str) or value is None:
        return value
    raise AskContextError("INTERNAL", "结果含有无法编码的值")
