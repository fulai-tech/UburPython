"""gRPC 访问日志：手板 / 量产 unary RPC 记开始、完成、耗时，并写入对应服务的控制层文件。

访问日志通过 bind(log_layer=controller) 进入该服务的 controller 文件。
服务包内除 rpc 以外的模块仍按模块名进入 service 文件，请求号用 contextualize 对齐。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable, Collection
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from loguru import logger

from app.core.logging import LAYER_CONTROLLER, rpc_access_logger

# 与 HTTP 请求日志对齐：限制单行摘要长度
MAX_SUMMARY_LEN = 2048
_LIST_PREVIEW = 3
_SERVICE_DIR = {
    "GenerationDataService": "facts",
}


class GrpcAccessLogInterceptor(grpc.aio.ServerInterceptor):
    """为 unary-unary RPC 记录访问日志；其它流式 handler 原样下放。"""

    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = await continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler
        method = handler_call_details.method or "unknown"
        return grpc.unary_unary_rpc_method_handler(
            _wrap_unary_unary(handler.unary_unary, method),
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def parse_rpc_method(method: str) -> tuple[str, str, str] | None:
    """把 /package.Service/Method 解析成 (产品, 服务目录, 方法名)。非手板/量产返回 None。"""
    if not method.startswith("/"):
        return None
    service_path, sep, rpc_name = method[1:].rpartition("/")
    if not sep or not rpc_name:
        return None
    package, sep, service_name = service_path.rpartition(".")
    if not sep or not service_name:
        return None
    if package == "uburnode.somni.v1":
        product = "somni"
    elif package == "uburnode.v1":
        product = "handboard"
    else:
        return None
    return product, _service_dir(service_name), rpc_name


def _service_dir(service_name: str) -> str:
    if service_name in _SERVICE_DIR:
        return _SERVICE_DIR[service_name]
    if service_name.endswith("Service"):
        return service_name[: -len("Service")].lower()
    return service_name.lower()


def _wrap_unary_unary(
    behavior: Callable[..., Awaitable[Any]],
    method: str,
) -> Callable[..., Awaitable[Any]]:
    route = parse_rpc_method(method)

    async def logged(request: Any, context: grpc.aio.ServicerContext) -> Any:
        request_id = _request_id(context)
        peer = _peer(context)
        started_at = time.perf_counter()
        access = logger
        token = None
        label = method
        if route is not None:
            product, service, rpc_name = route
            label = f"{product}.{service}.{rpc_name}"
            access = logger.bind(
                log_product=product,
                log_service=service,
                log_layer=LAYER_CONTROLLER,
            )
            token = rpc_access_logger.set(access)
        try:
            with logger.contextualize(request_id=request_id):
                access.info(
                    "gRPC 请求开始，接口={}，对端={}，请求={}",
                    label,
                    peer,
                    summarize_message(request),
                )
                try:
                    response = await behavior(request, context)
                except grpc.aio.AbortError:
                    access.info(
                        "gRPC 请求中止，接口={}，对端={}，状态={}，耗时={}毫秒",
                        label,
                        peer,
                        _status_name(context),
                        _elapsed_ms(started_at),
                    )
                    raise
                except Exception:
                    access.exception(
                        "gRPC 请求失败，接口={}，对端={}，耗时={}毫秒",
                        label,
                        peer,
                        _elapsed_ms(started_at),
                    )
                    raise
                access.info(
                    "gRPC 请求完成，接口={}，对端={}，耗时={}毫秒，响应={}",
                    label,
                    peer,
                    _elapsed_ms(started_at),
                    summarize_message(response),
                )
                return response
        finally:
            if token is not None:
                rpc_access_logger.reset(token)

    return logged


def summarize_message(message: Any) -> str:
    """把 protobuf 压成短摘要；大列表只保留预览，避免撑爆日志。"""
    if message is None:
        return ""
    if not isinstance(message, Message):
        return _truncate(str(message), MAX_SUMMARY_LEN)
    try:
        raw = MessageToDict(message, preserving_proto_field_name=True)
    except Exception:
        return type(message).__name__
    return _truncate(json.dumps(_shrink(raw), ensure_ascii=False), MAX_SUMMARY_LEN)


def _shrink(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) <= _LIST_PREVIEW:
            return [_shrink(item) for item in value]
        preview = [_shrink(item) for item in value[:_LIST_PREVIEW]]
        return {"_preview": preview, "_total": len(value)}
    if isinstance(value, dict):
        return {key: _shrink(item) for key, item in value.items()}
    return value


def _request_id(context: grpc.aio.ServicerContext) -> str:
    metadata = None
    try:
        metadata = context.invocation_metadata()
    except Exception:
        metadata = None
    if isinstance(metadata, Collection) and not isinstance(metadata, (str, bytes)):
        for item in metadata:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            key, value = item[0], item[1]
            if str(key).lower() != "x-request-id":
                continue
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="replace")
            else:
                text = str(value)
            text = text.strip()
            if text:
                return text[:128]
    return uuid.uuid4().hex[:16]


def _peer(context: grpc.aio.ServicerContext) -> str:
    try:
        return context.peer() or "unknown"
    except Exception:
        return "unknown"


def _status_name(context: grpc.aio.ServicerContext) -> str:
    try:
        code = context.code()
    except Exception:
        return "unknown"
    if code is None:
        return "unknown"
    return code.name if hasattr(code, "name") else str(code)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}…（已截断）"
