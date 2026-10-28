"""把 GetContext 的 query 组装成可交给 MongoDB find 的过滤条件。

客户端的 query 不会原样传入驱动。这里只抄写白名单字段，比较运算符和日期
$in 会换成新的字典；归属条件在最后写入，避免被请求体覆盖。
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

_MAX_REFS = 8
_MAX_STRING = 256
_MAX_IN_ITEMS = 366
_MAX_DEPTH = 6
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LOWER = frozenset({"$gt", "$gte"})
_UPPER = frozenset({"$lt", "$lte"})
_COMPARATORS = _LOWER | _UPPER

DEVICES_COLLECTION = "somni_devices"

# 根字段种类。create_time / update_time 不在四张表的字段表里，不进入白名单。
# object_id 目前拒绝查询：只约定对 _id 做 ObjectId 转换，字符串等值对不上库里的 ObjectId。
_SLEEP = {
    "uid": "string",
    "reports_id": "string",
    "record_date": "date_string",
    "session_id": "string",
    "language": "string",
    "sleep_summary": "document",
    "unified_data_timeline": "document",
    "ambient_climate_monitor": "document",
    "sleep_breakdown": "document",
    "sleep_schedule": "document",
    "overnight_vitals": "document",
    "daily_check": "document",
    "weather": "document",
    "ai_insight": "document",
    "ai_meeting_suggestions": "document",
}
_PLANS = {
    "plan_id": "string",
    "uid": "string",
    "device_id": "string",
    "date": "date_string",
    "version": "range_number",
    "name": "string",
    "language": "string",
    "phases": "document",
    "ai_analysis": "document",
    "ai_insights": "string",
    "week_day": "range_number",
}
_CLIPS = {
    "sn": "string",
    "clip_id": "string",
    "session_id": "string",
    "date": "date_string",
    "tag": "string",
    "ts": "range_number",
    "duration_ms": "range_number",
    "bytes": "range_number",
    "on_device": "bool",
    "file_id": "string",
    "sha256": "string",
    "public_url": "string",
    "uploaded_at": "range_date",
}
_WEARABLE = {
    "user_id": "string",
    "source": "string",
    "schema_version": "string",
    "status": "status",
    "timezone": "string",
    "record_date": "date_string",
    "activity_date": "date_string",
    "aggregation_version": "range_number",
    "aggregation_algorithm": "string",
    "input_sample_count": "range_number",
    "source_upload_record_ids": "object_id",
    "latest_upload_record_id": "object_id",
    "source_max_update_time": "range_date",
    "sleep_window": "document",
    "sleep": "document",
    "activity": "document",
    "physiology": "document",
    "data_gaps": "document",
    "profile_features": "document",
    "quality_flags": "document",
    "source_files": "document",
    "generated_at": "range_date",
}

TABLE_FIELDS: dict[str, dict[str, str]] = {
    "somni_sleep_reports": _SLEEP,
    "somni_plans": _PLANS,
    "somni_acoustic_clips": _CLIPS,
    "somni_wearable_daily_profiles": _WEARABLE,
}
_WEARABLE_TABLE = "somni_wearable_daily_profiles"
_CLIPS_TABLE = "somni_acoustic_clips"
_HIDDEN_READ_FIELDS = frozenset({"public_url"})


def readable_roots(table_name: str) -> frozenset[str]:
    if table_name not in TABLE_FIELDS:
        raise QueryBuildError("table_name 不在允许的四张表内")
    roots = set(TABLE_FIELDS[table_name]) - _HIDDEN_READ_FIELDS
    roots.add("_id")
    return frozenset(roots)


def document_roots(table_name: str) -> frozenset[str]:
    return frozenset(
        name for name, kind in TABLE_FIELDS[table_name].items() if kind == "document"
    )


class QueryBuildError(Exception):
    """查询条件无法组装。对应 gRPC INVALID_ARGUMENT。"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class PreparedQuery:
    index: int
    table_name: str
    collection: str
    mongo_filter: dict[str, Any]


@dataclass(frozen=True)
class AssemblyResult:
    """各表查询互不依赖，可以同时查。"""

    queries: tuple[PreparedQuery, ...]


def assemble_source_queries(source_refs: list[dict[str, Any]]) -> AssemblyResult:
    refs = _require_refs(source_refs)
    prepared: list[PreparedQuery] = []
    for index, ref in enumerate(refs):
        table_name = _require_table(ref)
        mongo_filter = _compile_query(table_name, ref.get("query"))
        prepared.append(
            PreparedQuery(
                index=index,
                table_name=table_name,
                collection=table_name,
                mongo_filter=mongo_filter,
            )
        )
    return AssemblyResult(queries=tuple(prepared))


def describe(value: Any) -> Any:
    """把 ObjectId、datetime 标成可读结构，方便查看组装结果。"""
    if isinstance(value, ObjectId):
        return {"$objectId": str(value)}
    if isinstance(value, datetime):
        return {"$date": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [describe(item) for item in value]
    return value


def _require_refs(source_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(source_refs, list):
        raise QueryBuildError("source_refs 必须是数组")
    if not 1 <= len(source_refs) <= _MAX_REFS:
        raise QueryBuildError("source_refs 必须是 1 到 8 项")
    for ref in source_refs:
        if not isinstance(ref, dict):
            raise QueryBuildError("source_refs 的每一项必须是对象")
    return source_refs


def _require_table(ref: dict[str, Any]) -> str:
    table_name = ref.get("table_name")
    if not isinstance(table_name, str) or table_name not in TABLE_FIELDS:
        raise QueryBuildError("table_name 不在允许的四张表内")
    return table_name


def _compile_query(table_name: str, query: Any) -> dict[str, Any]:
    if not isinstance(query, dict) or not query:
        raise QueryBuildError("query 必须是非空对象")
    fields = TABLE_FIELDS[table_name]
    compiled: dict[str, Any] = {}
    for key, value in query.items():
        if not isinstance(key, str) or not key or key.startswith("$") or "." in key:
            raise QueryBuildError(f"不允许的查询字段 {key!r}")
        if key == "_id":
            compiled["_id"] = _object_id(value)
            continue
        kind = fields.get(key)
        if kind is None:
            raise QueryBuildError(f"{table_name} 不允许查询字段 {key}")
        if kind == "object_id":
            raise QueryBuildError(f"{key} 是 ObjectId，当前只转换 _id，不能直接作为查询值")
        compiled[key] = _compile_value(kind, key, value)
        if table_name == _CLIPS_TABLE and key == "sn" and compiled["sn"] == "":
            raise QueryBuildError("sn 必须是非空字符串")
    _apply_wearable_status(table_name, compiled)
    return compiled


def _apply_wearable_status(table_name: str, compiled: dict[str, Any]) -> None:
    if table_name != _WEARABLE_TABLE:
        return
    status = compiled.get("status")
    if status is None:
        compiled["status"] = 1
    elif status != 1:
        raise QueryBuildError("status=0 的作废记录不能查询")


def _compile_value(kind: str, field: str, value: Any) -> Any:
    if kind == "string":
        return _plain_string(field, value)
    if kind == "date_string":
        return _date_string(field, value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise QueryBuildError(f"{field} 只能做布尔等值")
        return value
    if kind == "status":
        number = _coerce_number(value)
        if number == 0:
            raise QueryBuildError("status=0 的作废记录不能查询")
        if number != 1:
            raise QueryBuildError("status 只能等值查询，不能做范围比较")
        return 1
    if kind == "range_number":
        return _range_number(field, value)
    if kind == "range_date":
        return _range_date(field, value)
    if kind == "document":
        _reject_operators(value, field, depth=0)
        return copy.deepcopy(value)
    raise QueryBuildError(f"{field} 的字段规则未实现")


def _plain_string(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise QueryBuildError(f"{field} 只能做字符串等值")
    if len(value) > _MAX_STRING:
        raise QueryBuildError(f"{field} 超过 256 字符")
    return value


def _date_string(field: str, value: Any) -> str | dict[str, list[str]]:
    if isinstance(value, str):
        return _calendar_date(field, value)
    if isinstance(value, dict):
        if set(value) != {"$in"}:
            raise QueryBuildError(f"{field} 只允许等值或 $in")
        items = value["$in"]
        if not isinstance(items, list) or not items:
            raise QueryBuildError(f"{field} 的 $in 必须是非空数组")
        if len(items) > _MAX_IN_ITEMS:
            raise QueryBuildError(f"{field} 的 $in 超过 {_MAX_IN_ITEMS} 项")
        return {"$in": [_calendar_date(field, item) for item in items]}
    raise QueryBuildError(f"{field} 只允许等值或 $in")


def _calendar_date(field: str, value: Any) -> str:
    if not isinstance(value, str) or _DATE_RE.fullmatch(value) is None:
        raise QueryBuildError(f"{field} 必须是 YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise QueryBuildError(f"{field} 不是合法日期") from exc
    return value


def _range_number(field: str, value: Any) -> int | float | dict[str, Any]:
    number = _coerce_number(value)
    if number is not None:
        return number
    if isinstance(value, dict):
        return _comparators(field, value, _number_bound)
    raise QueryBuildError(f"{field} 只能做数值等值或范围比较")


def _range_date(field: str, value: Any) -> datetime | dict[str, Any]:
    if isinstance(value, str):
        return _aware_datetime(field, value)
    if isinstance(value, dict):
        return _comparators(field, value, _aware_datetime)
    raise QueryBuildError(f"{field} 只能做时间等值或范围比较")


def _comparators(field: str, value: dict[str, Any], parse_bound) -> dict[str, Any]:
    if not value:
        raise QueryBuildError(f"{field} 的比较条件不能为空")
    unknown = set(value) - _COMPARATORS
    if unknown:
        raise QueryBuildError(f"{field} 含有不允许的运算符 {sorted(unknown)}")
    lowers = [op for op in value if op in _LOWER]
    uppers = [op for op in value if op in _UPPER]
    if len(lowers) > 1 or len(uppers) > 1:
        raise QueryBuildError(f"{field} 的同一侧范围条件重复")
    parsed = {op: parse_bound(field, raw) for op, raw in value.items()}
    if lowers and uppers:
        lower_op = lowers[0]
        upper_op = uppers[0]
        if _empty_span(lower_op, parsed[lower_op], upper_op, parsed[upper_op]):
            raise QueryBuildError(f"{field} 的范围为空")
    return parsed


def _empty_span(lower_op: str, lower: Any, upper_op: str, upper: Any) -> bool:
    if lower > upper:
        return True
    if lower < upper:
        return False
    return lower_op == "$gt" or upper_op == "$lt"


def _number_bound(field: str, value: Any) -> int | float:
    number = _coerce_number(value)
    if number is None:
        raise QueryBuildError(f"{field} 的范围边界必须是有限数值")
    return number


def _coerce_number(value: Any) -> int | float | None:
    """Struct 里的整数会变成 1.0 这种浮点。整数值收回 int，数字字符串不转换。"""
    if type(value) is bool:
        return None
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        if value.is_integer():
            return int(value)
        return value
    return None


def _is_number(value: Any) -> bool:
    return _coerce_number(value) is not None


def _aware_datetime(field: str, value: Any) -> datetime:
    if not isinstance(value, str):
        raise QueryBuildError(f"{field} 必须是带时区的 ISO 8601 时间")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise QueryBuildError(f"{field} 必须是带时区的 ISO 8601 时间") from exc
    if parsed.tzinfo is None:
        raise QueryBuildError(f"{field} 必须带时区")
    return parsed


def _object_id(value: Any) -> ObjectId:
    if not isinstance(value, str):
        raise QueryBuildError("_id 必须是字符串")
    try:
        return ObjectId(value)
    except InvalidId as exc:
        raise QueryBuildError("_id 必须是 ObjectId") from exc


def _reject_operators(value: Any, field: str, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise QueryBuildError(f"{field} 嵌套过深")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key.startswith("$"):
                raise QueryBuildError(f"{field} 的等值条件里不能包含运算符")
            _reject_operators(item, field, depth + 1)
        return
    if isinstance(value, list):
        for item in value:
            _reject_operators(item, field, depth + 1)
        return
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise QueryBuildError(f"{field} 超过 256 字符")
        return
    if value is None or type(value) is bool or _is_number(value):
        if value is None:
            raise QueryBuildError(f"{field} 不支持 null")
        return
    raise QueryBuildError(f"{field} 含有不支持的值")
