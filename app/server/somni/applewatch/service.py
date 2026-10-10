"""量产 Apple Watch 按日聚合。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError
from app.server.somni.applewatch.store import AppleWatchStore
from app.server.somni.weather.fixed import fixed_weather

_FIXED_EMOTION = {
    "state": "Slightly Stressful",
    "stage": "Extended Unwind",
}

_FIXED_SCHEDULE_NAME = "Morning meeting"
_FIXED_SCHEDULE_START_AT = "9:00"  # 固定日程开始时刻（与 current_clock 同为 H:MM）

_TONIGHT_RISK_COPY = {
    "low": "LOW RISKS FOR TONIGHT",
    "medium": "MODERATE ACTIVITY TODAY",
    "excellent": "GREAT STEPS TODAY",
}


class AppleWatchService:
    def __init__(
        self,
        client: AsyncIOMotorClient | None,
        settings: Settings,
        store: AppleWatchStore | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or AppleWatchStore(client, settings)

    async def get_daily_aggregate(
        self,
        *,
        start_time: str,
        end_time: str,
        uid: str,
        device_id: str,
        timezone: str,
    ) -> dict[str, Any]:
        start_dt = parse_iso(start_time)
        end_dt = parse_iso(end_time)
        if end_dt <= start_dt:
            raise AppError(
                message="end_time 须晚于 start_time",
                status_code=HttpStatus.BAD_REQUEST,
            )
        tz = ZoneInfo(timezone)
        local_day = start_dt.astimezone(tz).date().isoformat()

        steps = await self._build_steps(
            uid=uid,
            timezone=timezone,
            start_dt=start_dt,
            end_dt=end_dt,
            display_date=local_day,
        )
        schedule = await self._build_schedule(
            uid=uid,
            device_id=device_id,
            timezone=timezone,
            anchor=start_dt,
        )
        return {
            "weather": fixed_weather(local_day),
            "steps": steps,
            "emotion_records": dict(_FIXED_EMOTION),
            "schedule": schedule,
        }

    async def _build_steps(
        self,
        *,
        uid: str,
        timezone: str,
        start_dt: datetime,
        end_dt: datetime,
        display_date: str,
    ) -> dict[str, Any]:
        docs = await self._store.list_step_points(
            uid=uid,
            timezone=timezone,
            start_at=_as_naive_utc(start_dt),
            end_at=_as_naive_utc(end_dt),
        )
        series: list[dict[str, Any]] = []
        total = 0
        seen: set[str] = set()
        fingerprints: dict[str, str] = {}
        tz = ZoneInfo(timezone)
        for doc in docs:
            sample_key = str(doc.get("sample_key") or "")
            fingerprint = _point_fingerprint(doc)
            if sample_key:
                if sample_key in fingerprints and fingerprints[sample_key] != fingerprint:
                    raise AppError(
                        message=f"Conflicting wearable sample: {sample_key}",
                        status_code=HttpStatus.CONFLICT,
                    )
                if sample_key in seen:
                    continue
                seen.add(sample_key)
                fingerprints[sample_key] = fingerprint
            value = int(float(doc.get("value_numeric") or 0))
            total += value
            series.append(
                {
                    "start_at": _format_local(doc.get("start_at"), tz),
                    "value": value,
                }
            )
        return {
            "date": display_date,
            "total": total,
            "series": series,
            "tonight_risk": tonight_risk_for_total(
                total,
                low=self._settings.applewatch_steps_level_low,
                medium=self._settings.applewatch_steps_level_medium,
            ),
        }

    async def _build_schedule(
        self,
        *,
        uid: str,
        device_id: str,
        timezone: str,
        anchor: datetime,
    ) -> dict[str, Any]:
        # 日程名称与开始时刻固定；current_clock 仍查 somni_alarms
        week_day = week_day_from_instant(anchor, timezone)
        alarm = await self._store.find_alarm(
            uid=uid, device_id=device_id, week_day=week_day
        )
        current_clock = "7:20"
        if alarm:
            wake = str(alarm.get("wake_time") or "").strip()
            if wake:
                current_clock = wake

        return {
            "schedule_name": _FIXED_SCHEDULE_NAME,
            "start_at": _FIXED_SCHEDULE_START_AT,
            "current_clock": current_clock,
        }


def parse_iso(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise AppError(
            message=f"时间格式非法: {value}",
            status_code=HttpStatus.BAD_REQUEST,
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def week_day_from_instant(value: datetime, tz_name: str) -> int:
    """周一=1 … 周日=7（按入参 timezone 的本地日）。"""
    local = value.astimezone(ZoneInfo(tz_name))
    return local.weekday() + 1


def week_day_from_record_date(record_date: str) -> int:
    """周一=1 … 周日=7。"""
    day = date.fromisoformat(record_date)
    return day.weekday() + 1


def tonight_risk_for_total(
    total: int, *, low: int = 5000, medium: int = 10000
) -> dict[str, str]:
    if total < low:
        level = "low"
    elif total < medium:
        level = "medium"
    else:
        level = "excellent"
    return {"level": level, "display_text": _TONIGHT_RISK_COPY[level]}


def _as_naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _format_local(value: Any, tz: ZoneInfo) -> str:
    dt = _as_datetime(value)
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).isoformat()


def _point_fingerprint(doc: dict[str, Any]) -> str:
    return "|".join(
        [
            str(doc.get("original_type") or ""),
            str(
                doc.get("value_numeric")
                if doc.get("value_numeric") is not None
                else doc.get("value_text") or ""
            ),
            str(doc.get("start_at") or ""),
            str(doc.get("end_at") or ""),
        ]
    )
