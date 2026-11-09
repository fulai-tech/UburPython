"""量产 Apple Watch gRPC 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.server.somni.weather.rpc import to_weather_message
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.applewatch.service import AppleWatchService


class AppleWatchRpc(uburnode_somni_pb2_grpc.AppleWatchServiceServicer):
    def __init__(self, service: AppleWatchService | None) -> None:
        self._service = service

    async def GetDailyAggregate(self, request, context):
        start_time = request.start_time.strip()
        end_time = request.end_time.strip()
        uid = request.uid.strip()
        device_id = request.device_id.strip()
        timezone = request.timezone.strip()
        if not start_time or not end_time or not uid or not device_id or not timezone:
            await abort_invalid(
                context, "start_time / end_time / uid / device_id / timezone 均不能为空"
            )
        try:
            ZoneInfo(timezone)
        except Exception:
            await abort_invalid(context, "timezone 非法")
        service = await self._require(context)

        async def _do():
            payload = await service.get_daily_aggregate(
                start_time=start_time,
                end_time=end_time,
                uid=uid,
                device_id=device_id,
                timezone=timezone,
            )
            return _to_res(payload)

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> AppleWatchService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def _to_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetDailyAggregateRes:
    steps = payload.get("steps") or {}
    risk = steps.get("tonight_risk") or {}
    emotion = payload.get("emotion_records") or {}
    schedule = payload.get("schedule") or {}
    res = uburnode_somni_pb2.GetDailyAggregateRes(
        weather=to_weather_message(payload.get("weather") or {}),
        steps=uburnode_somni_pb2.Steps(
            date=str(steps.get("date") or ""),
            total=int(steps.get("total") or 0),
            tonight_risk=uburnode_somni_pb2.TonightRisk(
                level=str(risk.get("level") or ""),
                display_text=str(risk.get("display_text") or ""),
            ),
        ),
        emotion_records=uburnode_somni_pb2.EmotionRecords(
            state=str(emotion.get("state") or ""),
            stage=str(emotion.get("stage") or ""),
        ),
        schedule=uburnode_somni_pb2.Schedule(
            schedule_name=str(schedule.get("schedule_name") or ""),
            start_at=str(schedule.get("start_at") or ""),
            current_clock=str(schedule.get("current_clock") or ""),
        ),
    )
    for item in steps.get("series") or []:
        res.steps.series.append(
            uburnode_somni_pb2.StepPoint(
                start_at=str(item.get("start_at") or ""),
                value=int(item.get("value") or 0),
            )
        )
    return res
