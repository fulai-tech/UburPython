"""量产问卷 gRPC 适配。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.quiz.service import QuizService


class QuizRpc(uburnode_somni_pb2_grpc.QuizServiceServicer):
    def __init__(self, service: QuizService | None) -> None:
        self._service = service

    async def GetAnswer(self, request, context):
        if not request.uid.strip():
            await abort_invalid(context, "uid 不能为空")
        service = await self._require(context)

        async def _do():
            payload = await service.get_answer(request.uid, request.answer_id)
            return _to_res(payload)

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> QuizService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def _to_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetAnswerRes:
    items = [
        {
            "question_id": str(item.get("question_id") or ""),
            "input_type": str(item.get("input_type") or ""),
            "title": str(item.get("title") or ""),
            "value": item.get("value"),
            "extra_input": str(item.get("extra_input") or ""),
        }
        for item in (payload.get("answers") or [])
        if isinstance(item, dict)
    ]
    return uburnode_somni_pb2.GetAnswerRes(
        answers=json.dumps(items, ensure_ascii=False, separators=(",", ":")),
    )
