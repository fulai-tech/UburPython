"""量产音频 gRPC 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.protobuf.struct_pb2 import Struct

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.audio.catalog import AudioCatalogService


class AudioRpc(uburnode_somni_pb2_grpc.AudioServiceServicer):
    def __init__(self, service: AudioCatalogService | None) -> None:
        self._service = service

    async def GetAudio(self, request, context):
        service = await self._require(context)

        async def _do():
            payload = await service.get_audio(**_get_audio_kwargs(request))
            return _to_audio_res(payload)

        return await run_rpc_call(context, _do)

    async def GetAudioTag(self, request, context):
        del request
        service = await self._require(context)

        async def _do():
            payload = await service.get_audio_tag()
            return _to_tag_res(payload)

        return await run_rpc_call(context, _do)

    async def GetHot(self, request, context):
        del request
        service = await self._require(context)

        async def _do():
            await service.get_hot()
            return uburnode_somni_pb2.GetHotRes()

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> AudioCatalogService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def _get_audio_kwargs(request) -> dict[str, Any]:
    return {
        "page": request.page if request.HasField("page") else None,
        "page_size": request.page_size if request.HasField("page_size") else None,
        "fetch_all": bool(request.fetch_all) if request.HasField("fetch_all") else False,
        "query_text": request.query_text if request.HasField("query_text") else "",
        "tag_code": request.tag_code if request.HasField("tag_code") else "",
    }


def _to_tag_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetAudioTagRes:
    res = uburnode_somni_pb2.GetAudioTagRes()
    for item in payload.get("tags") or []:
        res.tags.append(
            uburnode_somni_pb2.TagDictItem(
                type=str(item.get("type") or ""),
                code=str(item.get("code") or ""),
                name=str(item.get("name") or ""),
                name_en=str(item.get("name_en") or ""),
            )
        )
    return res


def _to_audio_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetAudioRes:
    res = uburnode_somni_pb2.GetAudioRes()
    for item in payload.get("materials") or []:
        struct = Struct()
        struct.update(item if isinstance(item, dict) else {})
        res.materials.append(struct)
    page = payload.get("page") or {}
    res.page.CopyFrom(
        uburnode_somni_pb2.PageInfo(
            page=int(page.get("page") or 1),
            page_size=int(page.get("page_size") or 0),
            total=int(page.get("total") or 0),
            total_pages=int(page.get("total_pages") or 0),
        )
    )
    return res
