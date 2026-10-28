"""GetContext：校验请求，并查四张表，按契约收成响应。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import Settings
from app.server.somni.ask.errors import AskContextError
from app.server.somni.ask.project import parse_filter_key, shape_table
from app.server.somni.ask.query_builder import (
    AssemblyResult,
    QueryBuildError,
    assemble_source_queries,
    readable_roots,
)
from app.server.somni.ask.store import AskContextStore

_RESPONSE_BYTES = 65536
_MAX_TOKEN = 256


class AskContextService:
    def __init__(self, client: AsyncIOMotorClient | None, settings: Settings) -> None:
        self._settings = settings
        self._store = AskContextStore(client, settings)

    @property
    def settings(self) -> Settings:
        return self._settings

    async def get_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise AskContextError("INVALID_ARGUMENT", "请求必须是对象")
        context_type = payload.get("context_type")
        query_id = _token(payload.get("query_id"), "query_id")
        if context_type == "session_history":
            result = await self._session_history(payload, query_id)
        elif context_type == "source_data":
            result = await self._source_data(payload, query_id)
        else:
            raise AskContextError(
                "INVALID_ARGUMENT",
                "context_type 必须是 source_data 或 session_history",
            )
        return _within_budget(result)

    async def _source_data(self, payload: dict[str, Any], query_id: str) -> dict[str, Any]:
        if payload.get("session_id"):
            raise AskContextError("INVALID_ARGUMENT", "source_data 不能带 session_id")
        refs = payload.get("source_refs")
        if not isinstance(refs, list):
            raise AskContextError("INVALID_ARGUMENT", "source_refs 必须是数组")
        try:
            assembled = assemble_source_queries(refs)
        except QueryBuildError as exc:
            raise AskContextError("INVALID_ARGUMENT", exc.message) from exc
        projections = [_projection(ref) for ref in refs]
        docs_by_index = await self._load(assembled)
        table_order: list[str] = []
        grouped: dict[str, list[tuple[list[dict[str, Any]], list[str] | None]]] = {}
        for index, ref in enumerate(refs):
            table_name = ref["table_name"]
            if table_name not in grouped:
                table_order.append(table_name)
                grouped[table_name] = []
            grouped[table_name].append((docs_by_index[index], projections[index]))
        table_data = {
            table_name: shape_table(grouped[table_name], readable_roots(table_name))
            for table_name in table_order
        }
        return {"query_id": query_id, "table_data": table_data}

    async def _load(self, assembled: AssemblyResult) -> dict[int, list[dict[str, Any]]]:
        queries = list(assembled.queries)
        if not queries:
            return {}
        found = await asyncio.gather(
            *(self._store.find(query.collection, query.mongo_filter) for query in queries)
        )
        return {query.index: docs for query, docs in zip(queries, found, strict=True)}

    async def _session_history(self, payload: dict[str, Any], query_id: str) -> dict[str, Any]:
        session_id = _token(payload.get("session_id"), "session_id")
        refs = payload.get("source_refs", [])
        if refs is None:
            refs = []
        if refs != []:
            raise AskContextError("INVALID_ARGUMENT", "session_history 不能带业务引用")
        messages = await self._store.find_chat_messages(session_id)
        return {
            "query_id": query_id,
            "session_id": session_id,
            "turns": _chat_turns(messages),
        }


def _projection(ref: dict[str, Any]) -> list[str] | None:
    table_name = ref.get("table_name")
    if not isinstance(table_name, str):
        raise AskContextError("INVALID_ARGUMENT", "table_name 不在允许的四张表内")
    return parse_filter_key(table_name, ref)


def _chat_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 message_index 把 user 与下一条 assistant 配成一轮。未完成的提问不能当作完整会话。"""
    pending: tuple[int, str] | None = None
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant" and pending is None:
            continue
        if role == "user":
            if pending is not None:
                raise AskContextError("FAILED_PRECONDITION", "对话尚未完整保存")
            pending = (_message_index(message), _message_text(message))
            continue
        if role == "assistant" and pending is not None:
            question_id, text_query = pending
            turns.append(
                {
                    "question_id": question_id,
                    "text_query": text_query,
                    "answer": _message_text(message),
                }
            )
            pending = None
            continue
        raise AskContextError("FAILED_PRECONDITION", "对话尚未完整保存")
    if pending is not None:
        raise AskContextError("FAILED_PRECONDITION", "对话尚未完整保存")
    return turns


def _message_index(message: dict[str, Any]) -> int:
    value = message.get("message_index")
    if type(value) is int:
        return value
    if type(value) is float and value.is_integer():
        return int(value)
    raise AskContextError("FAILED_PRECONDITION", "对话尚未完整保存")


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise AskContextError("FAILED_PRECONDITION", "对话尚未完整保存")


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > _MAX_TOKEN:
        raise AskContextError(
            "INVALID_ARGUMENT",
            f"{name} 必须是非空且不超过 256 字符的字符串",
        )
    return value


def _within_budget(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AskContextError("INTERNAL", "结果无法编码为 JSON") from exc
    if len(encoded) > _RESPONSE_BYTES:
        raise AskContextError("RESOURCE_EXHAUSTED", "结果超过 65536 字节")
    return payload
