"""量产睡眠报告：按 uid + record_date 聚合。"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
import math
from typing import Any

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError
from app.server.somni.report import calc
from app.server.somni.report.store import ReportStore
from app.server.somni.report.vitals_fixed import fixed_brv, fixed_hrv

_INTERVENTION_FIELDS = (
    "type",
    "event_time",
    "event_type",
    "duration",
    "trigger_cause",
    "action_taken",
    "result_summary",
)


class ReportService:
    def __init__(
        self,
        client: AsyncIOMotorClient | None,
        settings: Settings,
        store: ReportStore | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or ReportStore(client, settings)

    def _bucket_percent(self) -> float:
        return float(self._settings.somni_report_series_bucket_percent)

    async def get_summary(
        self, uid: str, record_date: str, session_id: str = ""
    ) -> dict[str, Any]:
        _ = uid, record_date
        on_bed, segments = await self._session_segments(session_id)
        structure = calc.sleep_structure_from_segments(on_bed, segments)
        docs = await self._store.list_telemetry_by_session(session_id, "sleep")
        return {
            "sleep_summary": {
                "body_battery": 0,
                "body_battery_status": "",
                "total_minutes": int(structure["total_minutes"]),
                "deep_sleep_minutes": int(structure["deep_sleep"]["minutes"]),
                "avg_heart_rate": calc.floor_avg(_data_floats(docs, "hr")),
                "avg_respiratory_rate": calc.floor_avg(_data_floats(docs, "br")),
            }
        }

    async def get_environment(
        self, uid: str, record_date: str, session_id: str = ""
    ) -> dict[str, Any]:
        _ = uid, record_date
        docs = await self._store.list_telemetry_by_session(session_id, "env")
        return {
            "environment_summary": {
                "temperature": calc.floor_metric_stats(_data_floats(docs, "temp")),
                "humidity": calc.floor_metric_stats(_data_floats(docs, "humi")),
                "illuminance": calc.floor_metric_stats(_data_floats(docs, "lux")),
                "noise": calc.floor_metric_stats(_data_floats(docs, "noise_db")),
            }
        }

    async def get_events(
        self, uid: str, record_date: str, session_id: str
    ) -> dict[str, Any]:
        events = await self._store.list_events(uid, record_date)
        sleep_events, abnormal_count, intervention_count = _assemble_sleep_events(events)
        on_bed, segments = await self._session_segments(session_id)
        sid = session_id.strip()
        sleep_docs = await self._store.list_telemetry_by_session(sid, "sleep")
        env_docs = await self._store.list_telemetry_by_session(sid, "env")
        percent = self._bucket_percent()
        return {
            "record_date": record_date,
            "sleep_events": sleep_events,
            "event_count": abnormal_count,
            "abnormal_count": abnormal_count,
            "intervention_count": intervention_count,
            "idf_data": calc.idf_from_segments(segments, on_bed),
            "physio_data": _physio_data(sleep_docs, percent),
            "env_data": _env_data(env_docs, percent),
        }

    async def get_structure(
        self,
        uid: str,
        record_date: str = "",
        start_date: str = "",
        end_date: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        sleep_structure = {
            "awake": {"minutes": 0, "percent": 0},
            "rem_sleep": {"minutes": 0, "percent": 0},
            "light_sleep": {"minutes": 0, "percent": 0},
            "deep_sleep": {"minutes": 0, "percent": 0},
        }
        if record_date.strip() or session_id.strip():
            on_bed, segments = await self._session_segments(session_id)
            if segments:
                parts = calc.sleep_structure_from_segments(on_bed, segments)
                sleep_structure = {
                    "awake": parts["awake"],
                    "rem_sleep": parts["rem_sleep"],
                    "light_sleep": parts["light_sleep"],
                    "deep_sleep": parts["deep_sleep"],
                }

        tz_name = calc.resolve_report_timezone_name(
            await self._store.find_user_timezone(uid)
        )
        week_start = start_date.strip()
        week_end = end_date.strip()
        if not week_start and not week_end:
            week_start, week_end = calc.default_week_bounds(timezone_name=tz_name)

        # 周报仍读 somni_records
        baseline_from = calc.baseline_start(week_end)
        docs = await self._store.list_records(uid, baseline_from, week_end)
        by_date: dict[str, dict[str, Any]] = {}
        for doc in docs:
            metric = calc.weekly_day_metric(doc)
            date_key = metric["record_date"]
            if date_key:
                by_date[date_key] = metric

        return {
            "sleep_structure": sleep_structure,
            "weekly_report": {
                "timezone": tz_name,
                "start_date": week_start,
                "end_date": week_end,
                "week_days": calc.align_weekly_days(by_date, week_start, week_end),
                "baseline_days": calc.align_weekly_days(
                    by_date, baseline_from, week_end
                ),
                "daytime_activity": {},
            },
        }

    async def get_profile(self, uid: str, record_date: str) -> dict[str, Any]:
        doc = await self._store.find_user_profile(uid, record_date)
        if doc is None:
            raise AppError(
                message=f"画像不存在：{uid}/{record_date}",
                status_code=HttpStatus.NOT_FOUND,
            )
        return {"profile_text": str(doc.get("profile_text") or "")}

    async def get_sleep_quality(
        self, uid: str, record_date: str, session_id: str = ""
    ) -> dict[str, Any]:
        _ = uid, record_date
        on_bed, segments = await self._session_segments(session_id)
        return {"sleep_quality": calc.sleep_quality_from_session(on_bed, segments)}

    async def get_vitals(
        self,
        *,
        uid: str = "",
        device_id: str = "",
        session_id: str = "",
        start_time: str = "",
        end_time: str = "",
    ) -> dict[str, Any]:
        query = await _build_vitals_query(
            self._store,
            uid=uid,
            device_id=device_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
        )
        docs = await self._store.list_telemetry_by_filter(query)
        hr = _avg_series(_metric_series(docs, "hr"))
        br = _avg_series(_metric_series(docs, "br"))
        hr["series"] = _half_hour_metric_series(docs, "hr")
        br["series"] = _half_hour_metric_series(docs, "br")
        return {
            "hr": hr,
            "br": br,
            "hrv": fixed_hrv(),
            "brv": fixed_brv(),
        }

    async def _session_segments(
        self, session_id: str
    ) -> tuple[Any, list[dict[str, Any]]]:
        doc = await self._store.find_session_payload(session_id)
        body = calc.session_body(doc)
        return calc.final_stages_segments(body)


def _data_floats(docs: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for doc in docs:
        data = doc.get("data") or {}
        if not isinstance(data, dict):
            continue
        num = calc.as_float(data.get(key))
        if num is not None:
            values.append(num)
    return values


def _physio_data(
    docs: list[dict[str, Any]], bucket_percent: float
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for start, bucket in _percent_groups(docs, bucket_percent):
        hrs = _bucket_floats(bucket, "hr")
        brs = _bucket_floats(bucket, "br")
        points.append(
            {
                "collected_at": calc.format_collected_at(start),
                "metrics": {
                    "heart_rate": _round_mean(hrs),
                    "respiration_rate": _round_mean(brs),
                },
            }
        )
    return points


def _env_data(
    docs: list[dict[str, Any]], bucket_percent: float
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for start, bucket in _percent_groups(docs, bucket_percent):
        points.append(
            {
                "collected_at": calc.format_collected_at(start),
                "temperature": _round_mean(_bucket_floats(bucket, "temp")),
                "humidity": _round_mean(_bucket_floats(bucket, "humi")),
                "illuminance": _round_mean(_bucket_floats(bucket, "lux")),
                "noise": _round_mean(_bucket_floats(bucket, "noise_db")),
            }
        )
    return points


_HALF_HOUR_SEC = 30 * 60


def _stamped_docs(
    docs: list[dict[str, Any]],
) -> list[tuple[datetime, dict[str, Any]]]:
    stamped: list[tuple[datetime, dict[str, Any]]] = []
    for doc in docs:
        ts = calc.coerce_datetime(doc.get("ts"))
        if ts is None:
            continue
        stamped.append((ts, doc))
    return stamped


def _percent_groups(
    docs: list[dict[str, Any]],
    bucket_percent: float,
) -> list[tuple[datetime, list[dict[str, Any]]]]:
    """按会话首尾时长的百分比分桶；docs 假定已按 ts 升序。"""
    stamped = _stamped_docs(docs)
    if not stamped:
        return []
    first_ts = stamped[0][0]
    last_ts = stamped[-1][0]
    total_sec = (last_ts - first_ts).total_seconds()
    if total_sec <= 0:
        return [(first_ts, [doc for _, doc in stamped])]

    fraction = max(float(bucket_percent), 1e-9) / 100.0
    bucket_sec = total_sec * fraction
    max_step = max(0, int(math.floor((total_sec - 1e-9) / bucket_sec)))
    groups: dict[datetime, list[dict[str, Any]]] = {}
    for ts, doc in stamped:
        offset = (ts - first_ts).total_seconds()
        steps = min(int(offset // bucket_sec), max_step)
        start = first_ts + timedelta(seconds=steps * bucket_sec)
        groups.setdefault(start, []).append(doc)
    return [(start, groups[start]) for start in sorted(groups)]


def _half_hour_groups(
    docs: list[dict[str, Any]],
) -> list[tuple[datetime, list[dict[str, Any]]]]:
    stamped = _stamped_docs(docs)
    if not stamped:
        return []
    anchor = min(ts for ts, _ in stamped)
    groups: dict[datetime, list[dict[str, Any]]] = {}
    for ts, doc in stamped:
        steps = int((ts - anchor).total_seconds() // _HALF_HOUR_SEC)
        start = anchor + timedelta(seconds=steps * _HALF_HOUR_SEC)
        groups.setdefault(start, []).append(doc)
    return [(start, groups[start]) for start in sorted(groups)]


def _bucket_floats(docs: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for doc in docs:
        data = doc.get("data")
        if not isinstance(data, dict):
            continue
        num = calc.as_float(data.get(key))
        if num is not None:
            values.append(num)
    return values


def _round_mean(values: list[float]) -> int:
    if not values:
        return 0
    total = sum((Decimal(str(value)) for value in values), Decimal(0))
    mean = total / Decimal(len(values))
    return int(mean.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _percent_metric_series(
    docs: list[dict[str, Any]], key: str, bucket_percent: float
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for start, bucket in _percent_groups(docs, bucket_percent):
        values = _bucket_floats(bucket, key)
        if not values:
            continue
        series.append(
            {
                "collected_at": calc.format_collected_at(start),
                "value": _round_mean(values),
            }
        )
    return series


def _half_hour_metric_series(
    docs: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for start, bucket in _half_hour_groups(docs):
        values = _bucket_floats(bucket, key)
        if not values:
            continue
        series.append(
            {
                "collected_at": calc.format_collected_at(start),
                "value": _round_mean(values),
            }
        )
    return series


def _assemble_sleep_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    abnormals: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    for doc in events:
        event_type = str(doc.get("type") or "")
        if event_type == "abnormal":
            abnormals.append(doc)
        elif event_type == "intervention":
            interventions.append(doc)

    by_id = {_event_id(doc): _to_sleep_event_item(doc) for doc in abnormals}
    used_parents: set[str] = set()
    for doc in interventions:
        parent_id = str(doc.get("related_event_id") or "")
        parent = by_id.get(parent_id)
        if parent is None:
            logger.debug("intervention 找不到父级 related_event_id={}", parent_id)
            continue
        if parent_id in used_parents:
            logger.debug("父级已有 intervention，忽略多余条 parent_id={}", parent_id)
            continue
        parent["intervention"] = _to_intervention(doc)
        used_parents.add(parent_id)

    return list(by_id.values()), len(abnormals), len(interventions)


def _event_id(doc: dict[str, Any]) -> str:
    return str(doc.get("_id") or doc.get("id") or "")


def _to_sleep_event_item(doc: dict[str, Any]) -> dict[str, Any]:
    details = []
    for item in doc.get("events") or []:
        if not isinstance(item, dict):
            continue
        details.append(
            {
                "event_type": str(item.get("event_type") or ""),
                "duration": str(item.get("duration") or ""),
                "trigger_cause": str(item.get("trigger_cause") or ""),
                "action_taken": str(item.get("action_taken") or ""),
                "result_summary": str(item.get("result_summary") or ""),
            }
        )
    return {
        "event_time": str(doc.get("event_time") or ""),
        "type": str(doc.get("type") or ""),
        "code": str(doc.get("code") or ""),
        "events": details,
        "intervention": {field: "" for field in _INTERVENTION_FIELDS},
    }


def _to_intervention(doc: dict[str, Any]) -> dict[str, str]:
    return {field: str(doc.get(field) or "") for field in _INTERVENTION_FIELDS}


async def _build_vitals_query(
    store: ReportStore,
    *,
    uid: str,
    device_id: str,
    session_id: str,
    start_time: str,
    end_time: str,
) -> dict[str, Any]:
    sid = session_id.strip()
    did = device_id.strip()
    user = uid.strip()
    start = start_time.strip()
    end = end_time.strip()
    has_window = bool(start and end)

    # 模式 1：有 session_id 且未给时间窗 → 按会话（其它字段忽略）
    if sid and not has_window:
        return {"session_id": sid, "metric": "sleep"}

    if not has_window:
        raise AppError(
            message="GetVitals 入参模式非法",
            status_code=HttpStatus.BAD_REQUEST,
        )

    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    if end_dt <= start_dt:
        raise AppError(
            message="end_time 须晚于 start_time",
            status_code=HttpStatus.BAD_REQUEST,
        )

    # 模式 2：uid + 时间窗，且未指定 device_id
    if user and not did:
        device_ids = await store.list_device_ids_by_uid(user)
        return {
            "device_id": {"$in": device_ids},
            "metric": "sleep",
            "ts": {"$gte": start_dt, "$lte": end_dt},
        }

    # 模式 3：device_id + 时间窗
    if did and not user:
        return {
            "device_id": did,
            "metric": "sleep",
            "ts": {"$gte": start_dt, "$lte": end_dt},
        }

    raise AppError(
        message="GetVitals 入参模式非法",
        status_code=HttpStatus.BAD_REQUEST,
    )


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _metric_series(docs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for doc in docs:
        data = doc.get("data") or {}
        if not isinstance(data, dict):
            continue
        num = calc.as_float(data.get(key))
        if num is None:
            continue
        points.append(
            {
                "collected_at": calc.format_collected_at(doc.get("ts")),
                "value": math.floor(num),
            }
        )
    return points


def _avg_series(series: list[dict[str, Any]]) -> dict[str, Any]:
    if not series:
        return {"value": 0, "series": []}
    values = [int(item["value"]) for item in series]
    return {"value": math.floor(sum(values) / len(values)), "series": series}
