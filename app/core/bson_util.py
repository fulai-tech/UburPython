"""BSON / 时间小工具（供手板、量产、同步脚本复用）。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from app.core.exceptions import MaterialNotFoundError


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def bson_to_jsonable(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: bson_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bson_to_jsonable(v) for v in value]
    return value


def parse_object_id(material_id: str) -> ObjectId:
    try:
        return ObjectId(material_id)
    except InvalidId as exc:
        raise MaterialNotFoundError(material_id) from exc
