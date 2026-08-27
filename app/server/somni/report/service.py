"""量产睡眠报告：按 uid + record_date 聚合。"""

from __future__ import annotations

from typing import Any

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import Settings
from app.server.somni.report import calc
from app.server.somni.report.store import ReportStore

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
        self._store = store or ReportStore(client, settings)

    async def get_summary(self, uid: str, record_date: str) -> dict[str, Any]:
        record = await self._store.find_record(uid, record_date)
        report = await self._store.find_sleep_report(uid, record_date)
        parts = calc.sleep_stage_parts(_raw_data(record))
        hr_vals, br_vals = await self._sleep_hr_br(uid, record_date)
        summary = (report or {}).get("sleep_summary") or {}
        return {
            "sleep_summary": {
                "body_battery": calc.as_int(summary.get("body_battery")),
                "body_battery_status": str(summary.get("body_battery_status") or ""),
                "total_minutes": int(parts["total_minutes"]),
                "deep_sleep_minutes": int(parts["deep_sleep"]["minutes"]),
                "avg_heart_rate": calc.floor_avg(hr_vals),
                "avg_respiratory_rate": calc.floor_avg(br_vals),
            }
        }

    async def get_environment(self, uid: str, record_date: str) -> dict[str, Any]:
        docs = await self._telemetry(uid, record_date, "env")
        return {
            "environment_summary": {
                "temperature": calc.floor_metric_stats(_data_floats(docs, "temp")),
                "humidity": calc.floor_metric_stats(_data_floats(docs, "humi")),
                "illuminance": calc.floor_metric_stats(_data_floats(docs, "lux")),
                "noise": calc.floor_metric_stats(_data_floats(docs, "noise_db")),
            }
        }

    async def get_events(self, uid: str, record_date: str) -> dict[str, Any]:
        events = await self._store.list_events(uid, record_date)
        sleep_events, abnormal_count, intervention_count = _assemble_sleep_events(events)
        record = await self._store.find_record(uid, record_date)
        sleep_docs = await self._telemetry(uid, record_date, "sleep")
        env_docs = await self._telemetry(uid, record_date, "env")
        return {
            "record_date": record_date,
            "sleep_events": sleep_events,
            "event_count": abnormal_count,
            "abnormal_count": abnormal_count,
            "intervention_count": intervention_count,
            "idf_data": _idf_data(record),
            "physio_data": _physio_data(sleep_docs),
            "env_data": _env_data(env_docs),
        }

    async def get_structure(self, uid: str, record_date: str) -> dict[str, Any]:
        record = await self._store.find_record(uid, record_date)
        parts = calc.sleep_stage_parts(_raw_data(record))
        return {
            "sleep_structure": {
                "awake": parts["awake"],
                "rem_sleep": parts["rem_sleep"],
                "light_sleep": parts["light_sleep"],
                "deep_sleep": parts["deep_sleep"],
            }
        }

    async def get_sleep_quality(self, uid: str, record_date: str) -> dict[str, Any]:
        record = await self._store.find_record(uid, record_date)
        raw = _raw_data(record)
        bed = raw.get("bed_time")
        wake = raw.get("wake_time")
        wake_up = raw.get("wake_up_time")
        return {
            "sleep_quality": {
                "time_in_bed_minutes": calc.minutes_between(bed, wake_up),
                "sleep_onset_latency_minutes": calc.as_int(raw.get("sleep_latency")),
                "sleep_efficiency": calc.as_int(raw.get("sleep_efficiency")),
                "bedtime": calc.format_hhmm(bed),
                "wake_up_time": calc.format_hhmm(wake_up),
                "awake_after_onset_minutes": calc.minutes_between(wake, wake_up),
            }
        }

    async def _telemetry(
        self, uid: str, record_date: str, metric: str
    ) -> list[dict[str, Any]]:
        device_id = await self._store.find_device_id(uid)
        if not device_id:
            return []
        return await self._store.list_telemetry(device_id, metric, record_date)

    async def _sleep_hr_br(
        self, uid: str, record_date: str
    ) -> tuple[list[float], list[float]]:
        docs = await self._telemetry(uid, record_date, "sleep")
        return _data_floats(docs, "hr"), _data_floats(docs, "br")


def _raw_data(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    raw = record.get("raw_data") or {}
    return raw if isinstance(raw, dict) else {}


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


def _idf_data(record: dict[str, Any] | None) -> list[dict[str, str]]:
    if not record:
        return []
    items = record.get("idf_data") or []
    result: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "stage": str(item.get("stage") or ""),
                "start": str(item.get("start") or ""),
                "end": str(item.get("end") or ""),
            }
        )
    return result


def _physio_data(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for doc in docs:
        data = doc.get("data") or {}
        hr = calc.as_float(data.get("hr") if isinstance(data, dict) else None)
        br = calc.as_float(data.get("br") if isinstance(data, dict) else None)
        points.append(
            {
                "collected_at": calc.format_collected_at(doc.get("ts")),
                "metrics": {
                    "heart_rate": 0 if hr is None else calc.floor_avg([hr]),
                    "respiration_rate": 0 if br is None else calc.floor_avg([br]),
                },
            }
        )
    return points


def _env_data(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for doc in docs:
        data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
        points.append(
            {
                "collected_at": calc.format_collected_at(doc.get("ts")),
                "temperature": _floor_one(data.get("temp")),
                "humidity": _floor_one(data.get("humi")),
                "illuminance": _floor_one(data.get("lux")),
                "noise": _floor_one(data.get("noise_db")),
            }
        )
    return points


def _floor_one(value: Any) -> int:
    num = calc.as_float(value)
    return 0 if num is None else calc.floor_avg([num])


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
