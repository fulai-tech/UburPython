"""量产 AskContext gRPC 适配。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct
from loguru import logger
from pymongo.errors import PyMongoError

from app.core.exceptions import MongoNotConfiguredError
from app.server.somni.ask.errors import AskContextError
from app.server.somni.ask.query_builder import QueryBuildError
from app.server.somni.ask.service import AskContextService
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2_grpc


def _brief_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    if len(text) > 300:
        return text[:300]
    return text


class AskContextRpc(uburnode_somni_pb2_grpc.AskContextServiceServicer):
    def __init__(self, service: AskContextService | None) -> None:
        self._service = service

    async def GetContext(self, request: Any, context: Any) -> Struct:
        started = time.perf_counter()
        query_id = ""
        context_type = ""
        status = "OK"
        reason = ""
        try:
            if self._service is None:
                raise AskContextError("UNAVAILABLE", "服务尚未就绪")
            payload = MessageToDict(request, preserving_proto_field_name=True)
            if isinstance(payload, dict):
                if isinstance(payload.get("query_id"), str):
                    query_id = payload["query_id"]
                if isinstance(payload.get("context_type"), str):
                    context_type = payload["context_type"]
            result = await asyncio.wait_for(
                self._service.get_context(payload),
                timeout=self._service.settings.ask_context_deadline_sec,
            )
            response = Struct()
            ParseDict(result, response)
            return response
        except QueryBuildError as exc:
            status = "INVALID_ARGUMENT"
            reason = exc.message
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, exc.message)
        except AskContextError as exc:
            status = exc.status
            reason = exc.message
            code = getattr(grpc.StatusCode, exc.status, grpc.StatusCode.INTERNAL)
            await context.abort(code, exc.message)
        except TimeoutError:
            status = "DEADLINE_EXCEEDED"
            reason = "GetContext 超时"
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, reason)
        except MongoNotConfiguredError:
            status = "UNAVAILABLE"
            reason = "Mongo 未配置"
            await context.abort(grpc.StatusCode.UNAVAILABLE, reason)
        except PyMongoError as exc:
            status = "UNAVAILABLE"
            reason = _brief_error(exc)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "查询失败")
        except Exception as exc:
            status = "INTERNAL"
            reason = _brief_error(exc)
            logger.exception("AskContext 处理失败 query_id={}", query_id)
            await context.abort(grpc.StatusCode.INTERNAL, "GetContext 失败")
        finally:
            log = logger.error if status != "OK" else logger.info
            log(
                "AskContext query_id={} context_type={} status={} reason={} ms={}",
                query_id,
                context_type,
                status,
                reason or "-",
                int((time.perf_counter() - started) * 1000),
            )
