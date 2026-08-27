"""功能手板 / 量产 QuizRpc。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from bson import ObjectId
from google.protobuf.json_format import MessageToDict

from app.core.config import Settings
from app.core.exceptions import AppError, MongoNotConfiguredError
from app.server.handboard.quiz.rpc import QuizRpc as HandboardQuizRpc
from app.server.somni.quiz.rpc import QuizRpc as SomniQuizRpc
from app.server.somni.quiz.service import QuizService
from app.uburnode_grpc.grpc_gen import uburnode_pb2, uburnode_somni_pb2


def _context() -> MagicMock:
    ctx = MagicMock()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError)
    return ctx


@pytest.mark.asyncio
async def test_handboard_get_answer_empty() -> None:
    rpc = HandboardQuizRpc()
    res = await rpc.GetAnswer(
        uburnode_pb2.GetAnswerReq(uid="u1", answer_id="a1"),
        _context(),
    )
    assert res == uburnode_pb2.GetAnswerRes()


@pytest.mark.asyncio
async def test_handboard_get_answer_requires_fields() -> None:
    rpc = HandboardQuizRpc()
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetAnswer(uburnode_pb2.GetAnswerReq(uid="", answer_id="a1"), _context())


@pytest.mark.asyncio
async def test_somni_get_answer_requires_fields() -> None:
    rpc = SomniQuizRpc(MagicMock())
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetAnswer(uburnode_somni_pb2.GetAnswerReq(), _context())


@pytest.mark.asyncio
async def test_somni_get_answer_maps_answer_item() -> None:
    service = MagicMock()
    service.get_answer = AsyncMock(
        return_value={
            "answers": [
                {
                    "question_id": "q1",
                    "input_type": "radio",
                    "title": "您的性别是？",
                    "tags": [],
                    "value": {"option_id": "A", "option_text": "先生"},
                    "extra_input": "",
                },
                {
                    "question_id": "q3",
                    "input_type": "checkbox",
                    "title": "关注阶段",
                    "value": [
                        {"option_id": "A", "option_text": "早期"},
                        {"option_id": "B", "option_text": "VC"},
                    ],
                    "extra_input": "",
                },
                {
                    "question_id": "q2",
                    "input_type": "input_number",
                    "title": "夜醒次数",
                    "value": 2,
                    "extra_input": "",
                },
                {
                    "question_id": "q4",
                    "input_type": "input",
                    "title": "补充说明",
                    "value": "最近压力比较大",
                    "extra_input": "",
                },
                {
                    "question_id": "q5",
                    "input_type": "switch",
                    "title": "是否启用",
                    "value": True,
                    "extra_input": "",
                },
            ]
        }
    )
    rpc = SomniQuizRpc(service)
    res = await rpc.GetAnswer(
        uburnode_somni_pb2.GetAnswerReq(uid="u1", answer_id="a1"),
        _context(),
    )
    payload = MessageToDict(res, preserving_proto_field_name=True)
    answers = json.loads(payload["answers"])
    assert len(answers) == 5
    assert answers[0] == {
        "question_id": "q1",
        "input_type": "radio",
        "title": "您的性别是？",
        "value": {"option_id": "A", "option_text": "先生"},
        "extra_input": "",
    }
    assert "tags" not in answers[0]
    assert answers[1]["value"] == [
        {"option_id": "A", "option_text": "早期"},
        {"option_id": "B", "option_text": "VC"},
    ]
    assert answers[2]["value"] == 2
    assert answers[3]["value"] == "最近压力比较大"
    assert answers[4]["value"] is True
    assert json.loads(res.answers)[0]["title"] == "您的性别是？"

@pytest.mark.asyncio
async def test_somni_quiz_service_loads_from_collection() -> None:
    doc = {
        "_id": "69b10cc516d7472aedf6bb80",
        "uid": "user-001",
        "answers": [
            {
                "question_id": "q1",
                "input_type": "radio",
                "title": "您的性别是？",
                "value": {"option_id": "A", "option_text": "先生"},
            },
            {
                "question_id": "q3",
                "input_type": "select",
                "title": "常驻城市",
                "value": {"option_id": "A", "option_text": "北京"},
            },
            {
                "question_id": "q4",
                "input_type": "checkbox",
                "title": "关注阶段",
                "value": [
                    {"option_id": "A", "option_text": "早期"},
                    {"option_id": "B", "option_text": "VC"},
                ],
            },
            {
                "question_id": "q2",
                "input_type": "input",
                "title": "补充说明",
                "value": "最近压力比较大",
                "extra_input": "备注",
            },
        ],
    }
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=doc)
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)

    settings = Settings(
        somni_mongo_db="Somni",
        somni_mongo_answers_collection="somni_quiz_answers",
    )
    svc = QuizService(client, settings)
    payload = await svc.get_answer("user-001", "69b10cc516d7472aedf6bb80")

    assert settings.somni_mongo_answers_collection == "somni_quiz_answers"
    client.__getitem__.assert_called_with("Somni")
    db.__getitem__.assert_called_with("somni_quiz_answers")
    assert payload["answers"][0]["input_type"] == "radio"
    assert payload["answers"][0]["value"] == {
        "option_id": "A",
        "option_text": "先生",
    }
    assert payload["answers"][1]["value"] == {
        "option_id": "A",
        "option_text": "北京",
    }
    assert payload["answers"][2]["value"] == [
        {"option_id": "A", "option_text": "早期"},
        {"option_id": "B", "option_text": "VC"},
    ]
    assert payload["answers"][3]["extra_input"] == "备注"
    assert "tags" not in payload["answers"][0]
    query = collection.find_one.await_args.args[0]
    assert query == {
        "uid": "user-001",
        "_id": ObjectId("69b10cc516d7472aedf6bb80"),
    }


@pytest.mark.asyncio
async def test_somni_quiz_service_not_found() -> None:
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)

    svc = QuizService(client, Settings(somni_mongo_answers_collection="somni_quiz_answers"))
    with pytest.raises(AppError) as exc:
        await svc.get_answer("u1", "missing-id")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_somni_quiz_service_requires_mongo() -> None:
    svc = QuizService(None, Settings())
    with pytest.raises(MongoNotConfiguredError):
        await svc.get_answer("u1", "a1")
