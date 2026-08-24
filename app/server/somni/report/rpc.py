"""量产睡眠报告 gRPC 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.report.service import ReportService

_RPC_RES = {
    "get_summary": uburnode_somni_pb2.GetSummaryRes,
    "get_events": uburnode_somni_pb2.GetEventsRes,
    "get_environment": uburnode_somni_pb2.GetEnvironmentRes,
    "get_structure": uburnode_somni_pb2.GetStructureRes,
    "get_sleep_quality": uburnode_somni_pb2.GetSleepQualityRes,
}


class ReportRpc(uburnode_somni_pb2_grpc.ReportServiceServicer):
    def __init__(self, service: ReportService | None) -> None:
        self._service = service

    async def GetSummary(self, request, context):
        return await self._call(request, context, "get_summary")

    async def GetEvents(self, request, context):
        return await self._call(request, context, "get_events")

    async def GetEnvironment(self, request, context):
        return await self._call(request, context, "get_environment")

    async def GetStructure(self, request, context):
        return await self._call(request, context, "get_structure")

    async def GetSleepQuality(self, request, context):
        return await self._call(request, context, "get_sleep_quality")

    async def _call(self, request, context, method_name: str):
        if not request.uid.strip() or not request.record_date.strip():
            await abort_invalid(context, "uid 与 record_date 均不能为空")
        service = await self._require(context)

        async def _do():
            await getattr(service, method_name)(request.uid, request.record_date)
            return _RPC_RES[method_name]()

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> ReportService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]
