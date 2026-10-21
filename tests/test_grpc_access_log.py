"""gRPC 访问日志：按服务写入控制层文件，服务层日志留在 service 文件。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from loguru import logger

from app.core.config import Settings
from app.core.logging import service_log_path, setup_logging
from app.server.grpc_logging import GrpcAccessLogInterceptor, parse_rpc_method
from app.server.somni.audio.catalog import AudioCatalogService
from app.server.somni.audio.rpc import AudioRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


def test_parse_rpc_method_maps_products_and_special_services() -> None:
    assert parse_rpc_method("/uburnode.somni.v1.AudioService/GetAudio") == (
        "somni",
        "audio",
        "GetAudio",
    )
    assert parse_rpc_method("/uburnode.somni.v1.GenerationDataService/GetPlanInputs") == (
        "somni",
        "facts",
        "GetPlanInputs",
    )
    assert parse_rpc_method("/uburnode.somni.v1.AppleWatchService/GetDailyAggregate") == (
        "somni",
        "applewatch",
        "GetDailyAggregate",
    )
    assert parse_rpc_method("/uburnode.v1.QuizService/GetAnswer") == (
        "handboard",
        "quiz",
        "GetAnswer",
    )
    reflection = "/grpc.reflection.v1alpha.ServerReflection/ServerReflectionInfo"
    assert parse_rpc_method(reflection) is None


@pytest.mark.asyncio
async def test_get_audio_access_log_is_split_from_service_log(tmp_path: Path) -> None:
    setup_logging(Settings(log_dir=str(tmp_path), log_level="INFO"))
    catalog = AudioCatalogService(None, Settings())

    async def behavior(request, context):
        await catalog._tag_ids_for_words([], ["雨"], None, "vector", 0.75)
        return uburnode_somni_pb2.GetAudioRes(page=1, page_size=20, total=0)

    handler = await _intercept(behavior, "/uburnode.somni.v1.AudioService/GetAudio")
    context = _context()
    await handler.unary_unary(
        uburnode_somni_pb2.GetAudioReq(query_text="雨声"),
        context,
    )
    logger.complete()

    day = date.today()
    controller = service_log_path(tmp_path, "somni", "audio", "controller", day).read_text(
        encoding="utf-8"
    )
    service = service_log_path(tmp_path, "somni", "audio", "service", day).read_text(
        encoding="utf-8"
    )
    common = (tmp_path / f"{day:%Y-%m-%d}_ubur_log").read_text(encoding="utf-8")

    assert "gRPC 请求开始" in controller
    assert "somni.audio.GetAudio" in controller
    assert "雨声" in controller
    assert "gRPC 请求完成" in controller
    assert "rid-audio-1" in controller
    assert "自由输入未做中文向量" in service
    assert "rid-audio-1" in service
    assert "gRPC 请求开始" not in service
    assert "自由输入未做中文向量" not in controller
    assert "gRPC 请求开始" not in common
    assert "自由输入未做中文向量" not in common


@pytest.mark.asyncio
async def test_rpc_failure_traceback_stays_in_controller_file(tmp_path: Path) -> None:
    setup_logging(Settings(log_dir=str(tmp_path), log_level="INFO"))
    service = MagicMock()
    service.get_audio = AsyncMock(side_effect=RuntimeError("目录炸了"))
    rpc = AudioRpc(service)

    async def behavior(request, context):
        return await rpc.GetAudio(request, context)

    handler = await _intercept(behavior, "/uburnode.somni.v1.AudioService/GetAudio")
    context = _context()
    context.abort = AsyncMock(side_effect=grpc.aio.AbortError)
    context.code.return_value = grpc.StatusCode.INTERNAL
    with pytest.raises(grpc.aio.AbortError):
        await handler.unary_unary(uburnode_somni_pb2.GetAudioReq(query_text="雨声"), context)
    logger.complete()

    day = date.today()
    controller = service_log_path(tmp_path, "somni", "audio", "controller", day).read_text(
        encoding="utf-8"
    )
    service_text = service_log_path(tmp_path, "somni", "audio", "service", day).read_text(
        encoding="utf-8"
    )
    assert "gRPC 处理失败" in controller
    assert "目录炸了" in controller
    assert "gRPC 请求中止" in controller
    assert "目录炸了" not in service_text


async def _intercept(behavior, method: str):
    interceptor = GrpcAccessLogInterceptor()

    async def continuation(_details):
        return grpc.unary_unary_rpc_method_handler(behavior)

    details = MagicMock()
    details.method = method
    return await interceptor.intercept_service(continuation, details)


def _context() -> MagicMock:
    context = MagicMock()
    context.peer.return_value = "ipv4:127.0.0.1:50000"
    context.invocation_metadata.return_value = (("x-request-id", "rid-audio-1"),)
    return context
