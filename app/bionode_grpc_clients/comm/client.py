"""comm-service gRPC 客户端（AudioMaterialService）。

UburNode 不直连 MongoDB；所有 CUD 经 comm-service（规范红线）。
proto 真源：仓库根 proto/bionode_comm.proto，变更后须重新 gen_proto.sh。
"""

from __future__ import annotations

import asyncio
from typing import Any

import grpc
from loguru import logger

from app.bionode_grpc_clients.comm.grpc_gen import bionode_comm_pb2, bionode_comm_pb2_grpc
from app.core.config import Settings
from app.schemas.audio import CreateAudioRequest, UpdateAudioRequest

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

    async def create_audio_material(self, request: CreateAudioRequest) -> None:
        stub = self._require_stub()
        await stub.CreateAudioMaterial(_to_create_req(request))

    async def update_audio_material(
        self, material_id: str, request: UpdateAudioRequest
    ) -> None:
        stub = self._require_stub()
        await stub.UpdateAudioMaterial(_to_update_req(material_id, request))

    async def delete_audio_material(self, material_id: str) -> None:
        stub = self._require_stub()
        await stub.DeleteAudioMaterial(bionode_comm_pb2.IdReq(id=material_id))

    async def list_audio_materials_page(
        self,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> tuple[list[bionode_comm_pb2.AudioMaterialInfo], int]:
        """分页拉取已发布原料；返回 (materials, total)。"""
        stub = self._require_stub()
        from app.bionode_grpc_clients.comm.grpc_gen import bionode_common_pb2

        response = await stub.ListAudioMaterials(
            bionode_comm_pb2.ListAudioMaterialsReq(
                page=bionode_common_pb2.PageRequest(
                    page=page,
                    page_size=page_size,
                    order_by="update_time desc",
                ),
                status=AUDIO_MATERIAL_STATUS_PUBLISHED,
            )
        )
        total = response.page.total if response.HasField("page") else len(response.materials)
        return list(response.materials), total

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


def _to_create_req(request: CreateAudioRequest) -> bionode_comm_pb2.CreateAudioMaterialReq:
    payload = request.model_dump(exclude_none=True)
    req = bionode_comm_pb2.CreateAudioMaterialReq(
        audio_name=request.audio_name,
        description=payload.get("description", ""),
        audio_url=payload.get("audio_url", ""),
        operation_type=int(payload.get("operation_type", 0)),
        created_by=payload.get("created_by", ""),
        updated_by=payload.get("updated_by", ""),
    )
    _copy_tag_fields(req, payload)
    _copy_embedding(req, payload)
    return req


def _to_update_req(
    material_id: str, request: UpdateAudioRequest
) -> bionode_comm_pb2.UpdateAudioMaterialReq:
    fields = request.model_dump(exclude_unset=True)
    req = bionode_comm_pb2.UpdateAudioMaterialReq(id=material_id)
    _set_optional_str(req, "description", fields)
    _set_optional_str(req, "audio_name", fields)
    _set_optional_str(req, "audio_url", fields)
    _set_optional_str(req, "created_by", fields)
    _set_optional_str(req, "updated_by", fields)
    if "operation_type" in fields and fields["operation_type"] is not None:
        req.operation_type = int(fields["operation_type"])
    if "status" in fields and fields["status"] is not None:
        req.status = 1 if fields["status"] else 0
    _copy_tag_fields(req, fields)
    _copy_embedding(req, fields)
    return req


def _set_optional_str(req: Any, field: str, fields: dict[str, Any]) -> None:
    if field not in fields or fields[field] is None:
        return
    setattr(req, field, str(fields[field]))


def _copy_tag_fields(req: Any, fields: dict[str, Any]) -> None:
    mapping = (
        "sleep_stage_tags",
        "content_form_tags",
        "mechanism_tags",
        "audio_engineering_tags",
        "medical_risk_tags",
        "evidence_level_tags",
    )
    for name in mapping:
        if name not in fields:
            continue
        getattr(req, name).extend(_to_proto_tags(fields[name] or []))


def _copy_embedding(req: Any, fields: dict[str, Any]) -> None:
    if "embedding" not in fields:
        return
    req.embedding[:] = [float(v) for v in fields["embedding"] or []]


def _to_proto_tags(tags: list[dict[str, Any]]) -> list[bionode_comm_pb2.AudioMaterialTag]:
    return [_to_proto_tag(tag) for tag in tags]


def _to_proto_tag(tag: dict[str, Any]) -> bionode_comm_pb2.AudioMaterialTag:
    msg = bionode_comm_pb2.AudioMaterialTag(
        tag_id=str(tag.get("tag_id") or ""),
        code=str(tag.get("code") or ""),
        name=str(tag.get("name") or ""),
    )
    if tag.get("en_name") is not None:
        msg.en_name = str(tag["en_name"])
    if tag.get("parent_tag_id") is not None:
        msg.parent_tag_id = str(tag["parent_tag_id"])
    if tag.get("parent_tag_code") is not None:
        msg.parent_tag_code = str(tag["parent_tag_code"])
    if tag.get("relative_loudness") is not None:
        msg.relative_loudness = float(tag["relative_loudness"])
    if tag.get("band_values"):
        msg.band_values[:] = [float(v) for v in tag["band_values"]]
    value = tag.get("value")
    if isinstance(value, dict):
        msg.value.tag_id = str(value.get("tag_id") or "")
        msg.value.code = str(value.get("code") or "")
        msg.value.name = str(value.get("name") or "")
    return msg
