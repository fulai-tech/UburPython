"""量产睡眠报告 gRPC 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.report.service import ReportService


class ReportRpc(uburnode_somni_pb2_grpc.ReportServiceServicer):
    def __init__(self, service: ReportService | None) -> None:
        self._service = service

    async def GetSummary(self, request, context):
        return await self._call(request, context, "get_summary", _to_summary_res)

    async def GetEvents(self, request, context):
        return await self._call(request, context, "get_events", _to_events_res)

    async def GetEnvironment(self, request, context):
        return await self._call(request, context, "get_environment", _to_environment_res)

    async def GetStructure(self, request, context):
        return await self._call(request, context, "get_structure", _to_structure_res)

    async def GetSleepQuality(self, request, context):
        return await self._call(
            request, context, "get_sleep_quality", _to_sleep_quality_res
        )

    async def GetProfile(self, request, context):
        return await self._call(request, context, "get_profile", _to_profile_res)

    async def _call(self, request, context, method_name: str, to_res):
        if not request.uid.strip() or not request.record_date.strip():
            await abort_invalid(context, "uid 与 record_date 均不能为空")
        service = await self._require(context)

        async def _do():
            payload = await getattr(service, method_name)(
                request.uid, request.record_date
            )
            return to_res(payload)

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> ReportService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def _to_profile_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetProfileRes:
    return uburnode_somni_pb2.GetProfileRes(
        profile_text=str(payload.get("profile_text") or "")
    )


def _to_summary_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetSummaryRes:
    item = payload.get("sleep_summary") or {}
    return uburnode_somni_pb2.GetSummaryRes(
        sleep_summary=uburnode_somni_pb2.SleepSummary(
            body_battery=int(item.get("body_battery") or 0),
            body_battery_status=str(item.get("body_battery_status") or ""),
            total_minutes=int(item.get("total_minutes") or 0),
            deep_sleep_minutes=int(item.get("deep_sleep_minutes") or 0),
            avg_heart_rate=int(item.get("avg_heart_rate") or 0),
            avg_respiratory_rate=int(item.get("avg_respiratory_rate") or 0),
        )
    )


def _to_environment_res(
    payload: dict[str, Any],
) -> uburnode_somni_pb2.GetEnvironmentRes:
    summary = payload.get("environment_summary") or {}
    return uburnode_somni_pb2.GetEnvironmentRes(
        environment_summary=uburnode_somni_pb2.EnvironmentSummary(
            temperature=_env_metric(summary.get("temperature")),
            humidity=_env_metric(summary.get("humidity")),
            illuminance=_env_metric(summary.get("illuminance")),
            noise=_env_metric(summary.get("noise")),
        )
    )


def _env_metric(item: Any) -> uburnode_somni_pb2.EnvMetric:
    data = item if isinstance(item, dict) else {}
    return uburnode_somni_pb2.EnvMetric(
        value=int(data.get("value") or 0),
        min=int(data.get("min") or 0),
        max=int(data.get("max") or 0),
    )


def _to_structure_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetStructureRes:
    structure = payload.get("sleep_structure") or {}
    return uburnode_somni_pb2.GetStructureRes(
        sleep_structure=uburnode_somni_pb2.SleepStructure(
            awake=_stage_part(structure.get("awake")),
            rem_sleep=_stage_part(structure.get("rem_sleep")),
            light_sleep=_stage_part(structure.get("light_sleep")),
            deep_sleep=_stage_part(structure.get("deep_sleep")),
        )
    )


def _stage_part(item: Any) -> uburnode_somni_pb2.SleepStagePart:
    data = item if isinstance(item, dict) else {}
    return uburnode_somni_pb2.SleepStagePart(
        minutes=int(data.get("minutes") or 0),
        percent=int(data.get("percent") or 0),
    )


def _to_sleep_quality_res(
    payload: dict[str, Any],
) -> uburnode_somni_pb2.GetSleepQualityRes:
    item = payload.get("sleep_quality") or {}
    return uburnode_somni_pb2.GetSleepQualityRes(
        sleep_quality=uburnode_somni_pb2.SleepQuality(
            time_in_bed_minutes=int(item.get("time_in_bed_minutes") or 0),
            sleep_onset_latency_minutes=int(
                item.get("sleep_onset_latency_minutes") or 0
            ),
            sleep_efficiency=int(item.get("sleep_efficiency") or 0),
            bedtime=str(item.get("bedtime") or ""),
            wake_up_time=str(item.get("wake_up_time") or ""),
            awake_after_onset_minutes=int(item.get("awake_after_onset_minutes") or 0),
        )
    )


def _to_events_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetEventsRes:
    res = uburnode_somni_pb2.GetEventsRes(
        record_date=str(payload.get("record_date") or ""),
        event_count=int(payload.get("event_count") or 0),
        abnormal_count=int(payload.get("abnormal_count") or 0),
        intervention_count=int(payload.get("intervention_count") or 0),
    )
    for item in payload.get("sleep_events") or []:
        res.sleep_events.append(_sleep_event_item(item))
    for item in payload.get("idf_data") or []:
        res.idf_data.append(
            uburnode_somni_pb2.IdfStage(
                stage=str(item.get("stage") or ""),
                start=str(item.get("start") or ""),
                end=str(item.get("end") or ""),
            )
        )
    for item in payload.get("physio_data") or []:
        metrics = item.get("metrics") or {}
        res.physio_data.append(
            uburnode_somni_pb2.PhysioDataPoint(
                collected_at=str(item.get("collected_at") or ""),
                metrics=uburnode_somni_pb2.PhysioMetrics(
                    heart_rate=int(metrics.get("heart_rate") or 0),
                    respiration_rate=int(metrics.get("respiration_rate") or 0),
                ),
            )
        )
    for item in payload.get("env_data") or []:
        res.env_data.append(
            uburnode_somni_pb2.EnvDataPoint(
                collected_at=str(item.get("collected_at") or ""),
                temperature=int(item.get("temperature") or 0),
                humidity=int(item.get("humidity") or 0),
                illuminance=int(item.get("illuminance") or 0),
                noise=int(item.get("noise") or 0),
            )
        )
    return res


def _sleep_event_item(item: dict[str, Any]) -> uburnode_somni_pb2.SleepEventItem:
    event = uburnode_somni_pb2.SleepEventItem(
        event_time=str(item.get("event_time") or ""),
        type=str(item.get("type") or ""),
        code=str(item.get("code") or ""),
    )
    for detail in item.get("events") or []:
        event.events.append(
            uburnode_somni_pb2.SleepEventDetail(
                event_type=str(detail.get("event_type") or ""),
                duration=str(detail.get("duration") or ""),
                trigger_cause=str(detail.get("trigger_cause") or ""),
                action_taken=str(detail.get("action_taken") or ""),
                result_summary=str(detail.get("result_summary") or ""),
            )
        )
    intervention = item.get("intervention") or {}
    event.intervention.CopyFrom(
        uburnode_somni_pb2.Intervention(
            type=str(intervention.get("type") or ""),
            event_time=str(intervention.get("event_time") or ""),
            event_type=str(intervention.get("event_type") or ""),
            duration=str(intervention.get("duration") or ""),
            trigger_cause=str(intervention.get("trigger_cause") or ""),
            action_taken=str(intervention.get("action_taken") or ""),
            result_summary=str(intervention.get("result_summary") or ""),
        )
    )
    return event
