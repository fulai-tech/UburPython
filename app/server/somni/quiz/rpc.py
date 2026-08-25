"""量产问卷 gRPC 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.protobuf.json_format import ParseDict

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.quiz.service import QuizService


class QuizRpc(uburnode_somni_pb2_grpc.QuizServiceServicer):
    def __init__(self, service: QuizService | None) -> None:
        self._service = service

    async def GetAnswer(self, request, context):
        if not request.uid.strip() or not request.answer_id.strip():
            await abort_invalid(context, "uid 与 answer_id 均不能为空")
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
    res = uburnode_somni_pb2.GetAnswerRes()
    for item in payload.get("answers") or []:
        res.answers.append(_to_item(item if isinstance(item, dict) else {}))
    return res


def _to_item(item: dict[str, Any]) -> uburnode_somni_pb2.AnswerItem:
    answer = uburnode_somni_pb2.AnswerItem(
        question_id=str(item.get("question_id") or ""),
        input_type=str(item.get("input_type") or ""),
        title=str(item.get("title") or ""),
        extra_input=str(item.get("extra_input") or ""),
    )
    raw = item.get("value")
    if raw is not None:
        ParseDict(raw, answer.value)
    return answer
