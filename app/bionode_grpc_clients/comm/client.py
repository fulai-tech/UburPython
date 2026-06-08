"""comm-service gRPC 客户端（AudioMaterialService）。

UburNode 不直连 MongoDB；所有 CUD 经 comm-service（规范红线）。
proto 真源：仓库根 proto/bionode_comm.proto，变更后须重新 gen_proto.sh。
"""

from __future__ import annotations

import asyncio

import grpc
from loguru import logger

from app.bionode_grpc_clients.comm.grpc_gen import bionode_comm_pb2, bionode_comm_pb2_grpc
from app.core.config import Settings
from app.schemas.audio import AudioMetaInfoIn

# comm ListAudioMaterials：status 默认 0 时列表为空；新建原料为已发布状态 1
AUDIO_MATERIAL_STATUS_PUBLISHED = 1


class CommClient:
    """封装 AudioMaterialService 的 gRPC 调用。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._channel: grpc.aio.Channel | None = None
        self._stub: bionode_comm_pb2_grpc.AudioMaterialServiceStub | None = None

    async def connect(self) -> None:
        target = self._settings.comm_grpc_target
        tls = self._settings.comm_grpc_use_tls
        logger.info("正在连接 comm-service gRPC：{}（TLS={}）", target, tls)
        if tls:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.aio.secure_channel(target, credentials)
        else:
            self._channel = grpc.aio.insecure_channel(target)
        self._stub = bionode_comm_pb2_grpc.AudioMaterialServiceStub(self._channel)

    async def ping(self, timeout_sec: float = 10.0) -> int:
        """探测 comm-service：调用 GetDistinctTags（首次 RPC 时建连）。"""
        if self._stub is None:
            raise RuntimeError("CommClient 未连接，请先调用 connect()")
        response = await asyncio.wait_for(
            self._stub.GetDistinctTags(bionode_comm_pb2.EmptyReq()),
            timeout=timeout_sec,
        )
        return len(response.tags)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    def _require_stub(self) -> bionode_comm_pb2_grpc.AudioMaterialServiceStub:
        if self._stub is None:
            raise RuntimeError("CommClient 未连接，请先在 lifespan 中调用 connect()")
        return self._stub

    async def get_audio_material(self, material_id: str) -> bionode_comm_pb2.AudioMaterialInfo:
        stub = self._require_stub()
        response = await stub.GetAudioMaterial(bionode_comm_pb2.IdReq(id=material_id))
        return response.material

    async def create_audio_material(
        self,
        *,
        category_code: int,
        noise_color: str | None,
        name: str,
        description: str,
        tags: list[str],
        audio_info: AudioMetaInfoIn,
    ) -> None:
        stub = self._require_stub()
        request = bionode_comm_pb2.CreateAudioMaterialReq(
            category_code=category_code,
            noise_color=noise_color or "",
            name=name,
            description=description,
            tags=tags,
            audio_info=_to_proto_audio_info(audio_info),
        )
        await stub.CreateAudioMaterial(request)

    async def update_audio_material(
        self,
        material_id: str,
        *,
        category_code: int,
        noise_color: str | None,
        name: str,
        description: str,
        tags: list[str],
        audio_info: AudioMetaInfoIn,
        status: int | None = None,
    ) -> None:
        """Update 需先 Get 再合并 status；其余字段以 HTTP 请求体为准。"""
        stub = self._require_stub()
        existing = await self.get_audio_material(material_id)

        request = bionode_comm_pb2.UpdateAudioMaterialReq(
            id=material_id,
            category_code=category_code,
            noise_color=noise_color or "",
            name=name,
            description=description,
            tags=tags,
            audio_info=_to_proto_audio_info(audio_info),
            status=status if status is not None else existing.status,
        )
        await stub.UpdateAudioMaterial(request)

    async def delete_audio_material(self, material_id: str) -> None:
        stub = self._require_stub()
        await stub.DeleteAudioMaterial(bionode_comm_pb2.IdReq(id=material_id))

    async def list_audio_materials_by_name(
        self, name: str
    ) -> list[bionode_comm_pb2.AudioMaterialInfo]:
        """Create 返回 EmptyRes 时的临时反查方案，待 proto 扩展后移除。"""
        stub = self._require_stub()
        from app.bionode_grpc_clients.comm.grpc_gen import bionode_common_pb2

        response = await stub.ListAudioMaterials(
            bionode_comm_pb2.ListAudioMaterialsReq(
                page=bionode_common_pb2.PageRequest(
                    page=1,
                    page_size=10,
                    order_by="create_time desc",
                ),
                name=name,
                status=AUDIO_MATERIAL_STATUS_PUBLISHED,
            )
        )
        return list(response.materials)


def _to_proto_audio_info(audio_info: AudioMetaInfoIn) -> bionode_comm_pb2.AudioMetaInfo:
    return bionode_comm_pb2.AudioMetaInfo(
        meta_data=bionode_comm_pb2.AudioMetaData(
            url=audio_info.meta_data.url,
            duration_sec=audio_info.meta_data.duration_sec,
        ),
        is_loopable=audio_info.is_loopable,
        is_voice=audio_info.is_voice,
    )
