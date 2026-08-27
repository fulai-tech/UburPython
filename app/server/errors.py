"""gRPC 错误映射与统一调用包装。"""

from __future__ import annotations

import grpc
from loguru import logger

from app.core.codes import HttpStatus
from app.core.exceptions import AppError, ElasticsearchUnavailableError

_STATUS_BY_HTTP: dict[int, grpc.StatusCode] = {
    HttpStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HttpStatus.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    HttpStatus.CONFLICT: grpc.StatusCode.ALREADY_EXISTS,
    HttpStatus.UNPROCESSABLE_ENTITY: grpc.StatusCode.INVALID_ARGUMENT,
    HttpStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    HttpStatus.BAD_GATEWAY: grpc.StatusCode.FAILED_PRECONDITION,
    412: grpc.StatusCode.FAILED_PRECONDITION,
    HttpStatus.GATEWAY_TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
    HttpStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
}


async def abort_invalid(context: grpc.aio.ServicerContext, message: str) -> None:
    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)


async def abort_from_app_error(
    context: grpc.aio.ServicerContext,
    exc: AppError,
) -> None:
    code = _STATUS_BY_HTTP.get(exc.status_code, grpc.StatusCode.INTERNAL)
    await context.abort(code, exc.message)


async def abort_internal(
    context: grpc.aio.ServicerContext,
    message: str = "服务器内部错误，请稍后重试",
) -> None:
    await context.abort(grpc.StatusCode.INTERNAL, message)


async def run_rpc_call(context: grpc.aio.ServicerContext, action):
    try:
        return await action()
    except grpc.aio.AbortError:
        raise
    except AppError as exc:
        await abort_from_app_error(context, exc)
    except Exception as exc:
        mapped = _maybe_es_error(exc)
        if mapped is not None:
            await abort_from_app_error(context, mapped)
        logger.exception("gRPC 处理失败：{}", exc)
        await abort_internal(context)


def _maybe_es_error(exc: Exception) -> AppError | None:
    try:
        from elastic_transport import TransportError
        from elasticsearch import ApiError
    except ImportError:  # pragma: no cover
        return None
    if isinstance(exc, (TransportError, ApiError)):
        return ElasticsearchUnavailableError(str(exc))
    return None
