"""功能手板音频 gRPC 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.server.handboard.audio import mapper as audio_mapper
from app.uburnode_grpc.grpc_gen import uburnode_pb2, uburnode_pb2_grpc

if TYPE_CHECKING:
    import grpc

    from app.server.handboard.audio.service import AudioService


class AudioRpc(uburnode_pb2_grpc.AudioServiceServicer):
    def __init__(self, audio_service: AudioService | None) -> None:
        self._audio = audio_service

    async def CreateAudio(self, request, context):
        service = await self._require_service(context)

        async def _do():
            body = audio_mapper.create_req_to_pydantic(request)
            data = await service.create_audio(body)
            return audio_mapper.dict_to_audio_material_res(data)

        return await run_rpc_call(context, _do)

    async def UpdateAudio(self, request, context):
        service = await self._require_service(context)

        async def _do():
            material_id, body = audio_mapper.update_req_to_pydantic(request)
            await service.update_audio(material_id, body)
            return audio_mapper.ok_operation("更新成功")

        return await run_rpc_call(context, _do)

    async def DeleteAudio(self, request, context):
        if not request.id.strip():
            await abort_invalid(context, "id 不能为空")
        service = await self._require_service(context)

        async def _do():
            await service.delete_audio(request.id)
            return audio_mapper.ok_operation("删除成功")

        return await run_rpc_call(context, _do)

    async def SearchAudio(self, request, context):
        service = await self._require_service(context)

        async def _do():
            body = audio_mapper.search_req_to_pydantic(request)
            data = await service.search_audio(body)
            return audio_mapper.search_data_to_res(data)

        return await run_rpc_call(context, _do)

    async def _require_service(self, context: grpc.aio.ServicerContext) -> AudioService:
        if self._audio is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._audio  # type: ignore[return-value]
