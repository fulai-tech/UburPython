"""双 gRPC 端口启停（功能手板 + 量产）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import grpc
from grpc_reflection.v1alpha import reflection
from loguru import logger

from app.server.handboard.audio.rpc import AudioRpc as HandboardAudioRpc
from app.server.handboard.quiz.rpc import QuizRpc as HandboardQuizRpc
from app.server.somni.audio.rpc import AudioRpc as SomniAudioRpc
from app.server.somni.profile.rpc import ProfileRpc as SomniProfileRpc
from app.server.somni.quiz.rpc import QuizRpc as SomniQuizRpc
from app.server.somni.report.rpc import ReportRpc as SomniReportRpc
from app.uburnode_grpc.grpc_gen import (
    uburnode_pb2,
    uburnode_pb2_grpc,
    uburnode_somni_pb2,
    uburnode_somni_pb2_grpc,
)

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.main import AppState


@dataclass
class GrpcServers:
    handboard: grpc.aio.Server | None = None
    somni: grpc.aio.Server | None = None


async def start_grpc_servers(state: AppState, settings: Settings) -> GrpcServers:
    servers = GrpcServers()
    if settings.grpc_enabled:
        servers.handboard = await _start_handboard(state, settings)
    else:
        logger.info("功能手板 gRPC 已关闭（GRPC_ENABLED=false）")
    if settings.somni_grpc_enabled:
        servers.somni = await _start_somni(state, settings)
    else:
        logger.info("量产 gRPC 已关闭（SOMNI_GRPC_ENABLED=false）")
    return servers


async def stop_grpc_servers(servers: GrpcServers | None, *, grace: float = 5.0) -> None:
    if servers is None:
        return
    for name, server in (("handboard", servers.handboard), ("somni", servers.somni)):
        if server is None:
            continue
        await server.stop(grace)
        logger.info("{} gRPC Server 已停止", name)


async def _start_handboard(state: AppState, settings: Settings) -> grpc.aio.Server:
    server = grpc.aio.server()
    uburnode_pb2_grpc.add_AudioServiceServicer_to_server(
        HandboardAudioRpc(getattr(state, "audio_service", None)),
        server,
    )
    uburnode_pb2_grpc.add_QuizServiceServicer_to_server(HandboardQuizRpc(), server)
    _enable_reflection(server, uburnode_pb2)
    bind = f"{settings.grpc_host}:{settings.grpc_port}"
    _bind(server, bind, "功能手板")
    await server.start()
    return server


async def _start_somni(state: AppState, settings: Settings) -> grpc.aio.Server:
    server = grpc.aio.server()
    uburnode_somni_pb2_grpc.add_QuizServiceServicer_to_server(
        SomniQuizRpc(getattr(state, "somni_quiz_service", None)),
        server,
    )
    uburnode_somni_pb2_grpc.add_ReportServiceServicer_to_server(
        SomniReportRpc(getattr(state, "somni_report_service", None)),
        server,
    )
    uburnode_somni_pb2_grpc.add_AudioServiceServicer_to_server(
        SomniAudioRpc(getattr(state, "somni_audio_service", None)),
        server,
    )
    uburnode_somni_pb2_grpc.add_ProfileServiceServicer_to_server(
        SomniProfileRpc(getattr(state, "somni_profile_service", None)),
        server,
    )
    _enable_reflection(server, uburnode_somni_pb2)
    bind = f"{settings.grpc_host}:{settings.somni_grpc_port}"
    _bind(server, bind, "量产")
    await server.start()
    return server


def _enable_reflection(server: grpc.aio.Server, proto_module) -> None:
    names = [reflection.SERVICE_NAME]
    names.extend(svc.full_name for svc in proto_module.DESCRIPTOR.services_by_name.values())
    reflection.enable_server_reflection(tuple(names), server)


def _bind(server: grpc.aio.Server, bind: str, label: str) -> None:
    if server.add_insecure_port(bind) == 0:
        raise RuntimeError(f"{label} gRPC 无法绑定 {bind}")
    logger.warning("{} gRPC 已以明文监听 {}（无鉴权；勿对公网直接暴露）", label, bind)
