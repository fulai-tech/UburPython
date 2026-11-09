"""量产用户画像 gRPC 适配。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.profile.service import ProfileService


class ProfileRpc(uburnode_somni_pb2_grpc.ProfileServiceServicer):
    def __init__(self, service: ProfileService | None) -> None:
        self._service = service

    async def GetUserProfile(self, request, context):
        if not request.uid.strip():
            await abort_invalid(context, "uid 不能为空")
        if not request.type.strip():
            await abort_invalid(context, "type 不能为空")
        profile_type = request.type.strip().lower()
        if profile_type in ("short_terms", "both") and not request.record_date.strip():
            await abort_invalid(context, "short_terms/both 类型必须传递 record_date")
        service = await self._require(context)

        async def _do():
            payload = await service.get_user_profile(
                request.uid.strip(),
                request.type.strip(),
                request.record_date.strip(),
            )
            return _to_res(payload)

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> ProfileService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def _to_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetUserProfileRes:
    return uburnode_somni_pb2.GetUserProfileRes(
        profile=json.dumps(
            payload.get("profile") or {},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
