"""功能手板问卷 gRPC 适配。"""

from __future__ import annotations

from app.server.errors import abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_pb2, uburnode_pb2_grpc


class QuizRpc(uburnode_pb2_grpc.QuizServiceServicer):
    async def GetAnswer(self, request, context):
        if not request.uid.strip() or not request.answer_id.strip():
            await abort_invalid(context, "uid 与 answer_id 均不能为空")

        async def _do():
            return uburnode_pb2.GetAnswerRes()

        return await run_rpc_call(context, _do)
