"""FastAPI 全局异常 → 统一 API 信封（code = HTTP 状态码）。"""

from __future__ import annotations

import grpc
from elastic_transport import TransportError
from elasticsearch import ApiError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.codes import HttpStatus
from app.core.exceptions import AppError, InternalError, UpstreamGrpcError
from app.core.response import json_fail


async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
    logger.warning("业务错误 status={} msg={}", exc.status_code, exc.message)
    return json_fail(exc.status_code, exc.message, status_code=exc.status_code)


async def handle_validation_error(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    msg = "; ".join(
        f"{'.'.join(str(p) for p in err.get('loc', []))}: {err.get('msg', '')}"
        for err in exc.errors()
    )
    return json_fail(
        HttpStatus.UNPROCESSABLE_ENTITY,
        msg or "请求参数校验失败",
        status_code=HttpStatus.UNPROCESSABLE_ENTITY,
    )


async def handle_http_exception(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "msg" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    message = detail if isinstance(detail, str) else str(detail)
    return json_fail(exc.status_code, message, status_code=exc.status_code)


async def handle_grpc_error(_request: Request, exc: grpc.aio.AioRpcError) -> JSONResponse:
    grpc_code = exc.code().name if exc.code() else "UNKNOWN"
    message = exc.details() or "comm-service 调用失败"
    app_exc = UpstreamGrpcError(grpc_code=grpc_code, message=message)
    logger.error("gRPC 失败 code={} details={}", grpc_code, message)
    return json_fail(app_exc.status_code, app_exc.message, status_code=app_exc.status_code)


async def handle_es_error(_request: Request, exc: Exception) -> JSONResponse:
    from app.core.exceptions import ElasticsearchUnavailableError

    app_exc = ElasticsearchUnavailableError(str(exc))
    logger.error("Elasticsearch 错误：{}", exc)
    return json_fail(app_exc.status_code, app_exc.message, status_code=app_exc.status_code)


async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理异常：{}", exc)
    app_exc = InternalError()
    return json_fail(
        HttpStatus.INTERNAL_SERVER_ERROR,
        app_exc.message,
        status_code=HttpStatus.INTERNAL_SERVER_ERROR,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(grpc.aio.AioRpcError, handle_grpc_error)
    app.add_exception_handler(TransportError, handle_es_error)
    app.add_exception_handler(ApiError, handle_es_error)
    app.add_exception_handler(Exception, handle_unexpected)
