"""量产音频 gRPC 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.audio.service import SomniAudioService


class AudioRpc(uburnode_somni_pb2_grpc.AudioServiceServicer):
    def __init__(self, service: SomniAudioService | None) -> None:
        self._service = service

    async def ListTags(self, request, context):
        service = await self._require(context)

        async def _do():
            data = await service.list_tags(
                page=request.page if request.HasField("page") else None,
                page_size=request.page_size if request.HasField("page_size") else None,
                fetch_all=bool(request.fetch_all) if request.HasField("fetch_all") else False,
                type_=request.type if request.HasField("type") else None,
                enabled_only=(
                    bool(request.enabled_only) if request.HasField("enabled_only") else False
                ),
                level=request.level if request.HasField("level") else 0,
            )
            return _list_tags_res(data)

        return await run_rpc_call(context, _do)

    async def ListAudios(self, request, context):
        service = await self._require(context)

        async def _do():
            data = await service.list_audios(
                page=request.page if request.HasField("page") else None,
                page_size=request.page_size if request.HasField("page_size") else None,
                fetch_all=bool(request.fetch_all) if request.HasField("fetch_all") else False,
                enabled_only=(
                    bool(request.enabled_only) if request.HasField("enabled_only") else False
                ),
                tags=list(request.tags),
            )
            return _list_audios_res(data)

        return await run_rpc_call(context, _do)

    async def SearchAudio(self, request, context):
        if not request.query_text.strip():
            await abort_invalid(context, "query_text 不能为空")
        service = await self._require(context)

        async def _do():
            top_k = request.top_k if request.HasField("top_k") else None
            data = await service.search_audio(request.query_text, top_k)
            res = uburnode_somni_pb2.SearchAudioRes()
            for item in data.materials:
                struct = Struct()
                struct.update(item if isinstance(item, dict) else {})
                res.materials.append(struct)
            return res

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> SomniAudioService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def _list_tags_res(data: dict[str, Any]) -> uburnode_somni_pb2.ListTagsRes:
    res = uburnode_somni_pb2.ListTagsRes()
    for tag in data.get("tags", []):
        msg = uburnode_somni_pb2.Tag()
        ParseDict(tag, msg, ignore_unknown_fields=True)
        res.tags.append(msg)
    page = data.get("page") or {}
    res.page.CopyFrom(
        uburnode_somni_pb2.PageInfo(
            page=int(page.get("page") or 1),
            page_size=int(page.get("page_size") or 0),
            total=int(page.get("total") or 0),
            total_pages=int(page.get("total_pages") or 0),
        )
    )
    return res


def _list_audios_res(data: dict[str, Any]) -> uburnode_somni_pb2.ListAudiosRes:
    res = uburnode_somni_pb2.ListAudiosRes()
    for material in data.get("materials", []):
        msg = uburnode_somni_pb2.AudioMaterial()
        ParseDict(material, msg, ignore_unknown_fields=True)
        res.materials.append(msg)
    page = data.get("page") or {}
    res.page.CopyFrom(
        uburnode_somni_pb2.PageInfo(
            page=int(page.get("page") or 1),
            page_size=int(page.get("page_size") or 0),
            total=int(page.get("total") or 0),
            total_pages=int(page.get("total_pages") or 0),
        )
    )
    return res
