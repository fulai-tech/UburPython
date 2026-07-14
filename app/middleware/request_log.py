"""HTTP 请求日志中间件。

每次请求写入 logs/uburnode.log：入站 / 出站 / 耗时 / 状态码 / 请求体。
使用 @app.middleware("http") 而非 BaseHTTPMiddleware，避免异常绕过 FastAPI handler。
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request, Response
from loguru import logger
from starlette.types import Message

MAX_LOG_BODY_LEN = 2048


def register_request_log_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_log_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        client_host = request.client.host if request.client else "unknown"
        started_at = time.perf_counter()
        body_text = await _read_and_restore_body(request)

        bound = logger.bind(request_id=request_id)
        bound.info(
            "请求开始，方法={}，路径={}，查询参数={}，请求体={}，客户端={}",
            request.method,
            request.url.path,
            _truncate(request.url.query),
            _truncate(body_text),
            client_host,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = _elapsed_ms(started_at)
            bound.exception(
                "请求失败，方法={}，路径={}，耗时={}毫秒",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = _elapsed_ms(started_at)
        bound.info(
            "请求完成，方法={}，路径={}，状态码={}，耗时={}毫秒",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-Id"] = request_id
        return response


async def _read_and_restore_body(request: Request) -> str:
    """读 body 供日志，并写回 receive，避免下游拿不到请求体。"""
    body = await request.body()
    await _replay_body(request, body)
    if not body:
        return ""
    return body.decode("utf-8", errors="replace")


async def _replay_body(request: Request, body: bytes) -> None:
    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive  # noqa: SLF001 — Starlette 标准重放做法


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _truncate(value: str, max_len: int = MAX_LOG_BODY_LEN) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}…（已截断）"
