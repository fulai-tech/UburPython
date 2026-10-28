"""gRPC 访问日志：按服务写入控制层文件，服务层日志留在 service 文件。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from loguru import logger
from pymongo.errors import ServerSelectionTimeoutError

from app.core.config import Settings
from app.core.logging import setup_logging
from app.server.grpc_logging import GrpcAccessLogInterceptor, parse_rpc_method
from app.server.somni.audio.catalog import AudioCatalogService
from app.server.somni.audio.rpc import AudioRpc
from app.server.somni.facts.rpc import GenerationDataRpc
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


@pytest.mark.asyncio
async def test_ask_access_log_omits_request_and_response_body(tmp_path: Path) -> None:
    setup_logging(Settings(log_dir=str(tmp_path), log_level="INFO"))

    request = uburnode_somni_pb2.GetContextReq(
        context_type="source_data",
        query_id="ask-trace-1",
        source_refs=[
            uburnode_somni_pb2.AskContextSourceRef(
                table_name="somni_sleep_reports",
                query='{"record_date":"2026-10-01","uid":"secret-user"}',
            )
        ],
    )

    async def behavior(req, context):
        del req, context
        response = uburnode_somni_pb2.GetContextRes(query_id="ask-trace-1")
        response.table_data["somni_sleep_reports"] = '{"sleep_summary":{"body_battery":80}}'
        return response

    handler = await _intercept(behavior, "/uburnode.somni.v1.AskContextService/GetContext")
    await handler.unary_unary(request, _context())
    logger.complete()

    text = (tmp_path / "somni" / "ask" / f"{date.today():%Y-%m-%d}_log").read_text(
        encoding="utf-8"
    )
    assert "query_id=ask-trace-1" in text
    assert "context_type=source_data" in text
    assert "secret-user" not in text
    assert "sleep_summary" not in text
    assert "body_battery" not in text
    assert "record_date" not in text


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
    assert parse_rpc_method("/uburnode.somni.v1.AskContextService/GetContext") == (
        "somni",
        "ask",
        "GetContext",
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
    text = (tmp_path / "somni" / "audio" / f"{day:%Y-%m-%d}_log").read_text(encoding="utf-8")
    common = (tmp_path / f"{day:%Y-%m-%d}_ubur_log").read_text(encoding="utf-8")

    assert "gRPC 请求开始" in text
    assert "somni.audio.GetAudio" in text
    assert "雨声" in text
    assert "gRPC 请求完成" in text
    assert "rid-audio-1" in text
    assert "自由输入未做中文向量" in text
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
    context = _aborting_context()
    with pytest.raises(grpc.aio.AbortError):
        await handler.unary_unary(uburnode_somni_pb2.GetAudioReq(query_text="雨声"), context)
    logger.complete()

    day = date.today()
    text = (tmp_path / "somni" / "audio" / f"{day:%Y-%m-%d}_log").read_text(encoding="utf-8")
    assert "gRPC 处理失败" in text
    assert "目录炸了" in text
    assert "gRPC 请求中止" in text
    assert "状态=INTERNAL" in text
    assert "原因=服务器内部错误，请稍后重试" in text


@pytest.mark.asyncio
async def test_abort_log_includes_precondition_reason(tmp_path: Path) -> None:
    setup_logging(Settings(log_dir=str(tmp_path), log_level="INFO"))

    async def behavior(request, context):
        del request
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "active_plan_link_incomplete")

    handler = await _intercept(
        behavior, "/uburnode.somni.v1.GenerationDataService/GetPlanInputs"
    )
    context = _aborting_context()
    with pytest.raises(grpc.aio.AbortError):
        await handler.unary_unary(object(), context)
    logger.complete()

    text = _facts_log(tmp_path)
    assert "状态=FAILED_PRECONDITION" in text
    assert "原因=active_plan_link_incomplete" in text


@pytest.mark.asyncio
async def test_generation_mongo_error_keeps_cause_in_controller_log(tmp_path: Path) -> None:
    setup_logging(Settings(log_dir=str(tmp_path), log_level="INFO"))
    service = MagicMock()
    service.settings = Settings()
    service.get_plan_inputs = AsyncMock(
        side_effect=ServerSelectionTimeoutError("localhost:27017: Connection refused")
    )
    rpc = GenerationDataRpc(service)

    async def behavior(request, context):
        return await rpc.GetPlanInputs(request, context)

    handler = await _intercept(
        behavior, "/uburnode.somni.v1.GenerationDataService/GetPlanInputs"
    )
    with pytest.raises(grpc.aio.AbortError):
        await handler.unary_unary(
            uburnode_somni_pb2.GetPlanInputsReq(uid="user-1", request_id="req-1"),
            _aborting_context(),
        )
    logger.complete()

    text = _facts_log(tmp_path)
    assert "gRPC 处理失败" in text
    assert "Connection refused" in text
    assert "Traceback" in text
    assert "rid-audio-1" in text
    assert "状态=UNAVAILABLE" in text
    assert "原因=source_unavailable" in text
    common = (tmp_path / f"{date.today():%Y-%m-%d}_ubur_log").read_text(encoding="utf-8")
    assert "Connection refused" not in common


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status", "reason", "needle"),
    [
        (TimeoutError(), "DEADLINE_EXCEEDED", "generation_deadline_exceeded", "TimeoutError"),
        (RuntimeError("生成计算炸了"), "INTERNAL", "generation_request_failed", "生成计算炸了"),
        (
            ValueError("mismatched_intervals"),
            "INVALID_ARGUMENT",
            "invalid_request",
            "mismatched_intervals",
        ),
    ],
)
async def test_generation_hidden_errors_are_logged_with_original_cause(
    tmp_path: Path,
    error: Exception,
    status: str,
    reason: str,
    needle: str,
) -> None:
    text = await _generation_failure_log(tmp_path, error)
    assert "gRPC 处理失败" in text
    assert needle in text
    assert "Traceback" in text
    assert f"状态={status}" in text
    assert f"原因={reason}" in text


@pytest.mark.asyncio
async def test_generation_contract_violation_logs_parse_cause(tmp_path: Path) -> None:
    setup_logging(Settings(log_dir=str(tmp_path), log_level="INFO"))
    service = MagicMock()
    service.settings = Settings()
    service.get_plan_inputs = AsyncMock(return_value={"read_started_at_utc": "not-a-timestamp"})
    rpc = GenerationDataRpc(service)

    async def behavior(request, context):
        return await rpc.GetPlanInputs(request, context)

    handler = await _intercept(
        behavior, "/uburnode.somni.v1.GenerationDataService/GetPlanInputs"
    )
    with pytest.raises(grpc.aio.AbortError):
        await handler.unary_unary(
            uburnode_somni_pb2.GetPlanInputsReq(uid="user-1", request_id="req-1"),
            _aborting_context(),
        )
    logger.complete()

    text = _facts_log(tmp_path)
    assert "gRPC 处理失败" in text
    assert "response_contract_violation" in text
    assert "read_started_at_utc" in text
    assert "Traceback" in text
    assert "状态=INTERNAL" in text
    assert "原因=response_contract_violation" in text


@pytest.mark.asyncio
async def test_generation_invalid_uid_stays_a_single_abort_line(tmp_path: Path) -> None:
    setup_logging(Settings(log_dir=str(tmp_path), log_level="INFO"))
    rpc = GenerationDataRpc(MagicMock())

    async def behavior(request, context):
        return await rpc.GetPlanInputs(request, context)

    handler = await _intercept(
        behavior, "/uburnode.somni.v1.GenerationDataService/GetPlanInputs"
    )
    with pytest.raises(grpc.aio.AbortError):
        await handler.unary_unary(
            uburnode_somni_pb2.GetPlanInputsReq(uid="", request_id="req-1"),
            _aborting_context(),
        )
    logger.complete()

    text = _facts_log(tmp_path)
    assert "gRPC 处理失败" not in text
    assert "Traceback" not in text
    assert "状态=INVALID_ARGUMENT" in text
    assert "原因=invalid_uid" in text


async def _generation_failure_log(log_dir: Path, error: Exception) -> str:
    setup_logging(Settings(log_dir=str(log_dir), log_level="INFO"))
    service = MagicMock()
    service.settings = Settings()
    service.get_plan_inputs = AsyncMock(side_effect=error)
    rpc = GenerationDataRpc(service)

    async def behavior(request, context):
        return await rpc.GetPlanInputs(request, context)

    handler = await _intercept(
        behavior, "/uburnode.somni.v1.GenerationDataService/GetPlanInputs"
    )
    with pytest.raises(grpc.aio.AbortError):
        await handler.unary_unary(
            uburnode_somni_pb2.GetPlanInputsReq(uid="user-1", request_id="req-1"),
            _aborting_context(),
        )
    logger.complete()
    return _facts_log(log_dir)


def _facts_log(log_dir: Path) -> str:
    return (log_dir / "somni" / "facts" / f"{date.today():%Y-%m-%d}_log").read_text(
        encoding="utf-8"
    )


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
    context.details.return_value = ""
    return context


def _aborting_context() -> MagicMock:
    context = _context()

    async def abort(code, details="", trailing_metadata=()):
        del trailing_metadata
        context.code.return_value = code
        context.details.return_value = details
        raise grpc.aio.AbortError

    context.abort = abort
    return context
