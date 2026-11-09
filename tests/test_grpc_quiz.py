"""功能手板 / 量产 QuizRpc。"""

from __future__ import annotations

import json
from typing import Any
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


def _db_collections(answers: MagicMock, questions: list | None = None) -> Any:
    question_cursor = MagicMock()
    question_cursor.to_list = AsyncMock(return_value=questions or [])
    question_collection = MagicMock()
    question_collection.find = MagicMock(return_value=question_cursor)

    def _get(name: str) -> MagicMock:
        if "question" in name:
            return question_collection
        return answers

    return _get


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
async def test_somni_get_answer_requires_uid() -> None:
    rpc = SomniQuizRpc(MagicMock())
    with pytest.raises(grpc.aio.AbortError):
        await rpc.GetAnswer(uburnode_somni_pb2.GetAnswerReq(), _context())


@pytest.mark.asyncio
async def test_somni_get_answer_allows_empty_answer_id() -> None:
    service = MagicMock()
    service.get_answer = AsyncMock(return_value={"answers": []})
    rpc = SomniQuizRpc(service)
    await rpc.GetAnswer(
        uburnode_somni_pb2.GetAnswerReq(uid="u1", answer_id=""),
        _context(),
    )
    service.get_answer.assert_awaited_once_with("u1", "")


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
    db.__getitem__ = MagicMock(side_effect=_db_collections(collection))
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
    db.__getitem__.assert_any_call("somni_quiz_answers")
    assert payload["answers"][0]["input_type"] == "radio"
    assert payload["answers"][0]["value"] == "先生"
    assert payload["answers"][1]["value"] == "北京"
    assert payload["answers"][2]["value"] == ["早期", "VC"]
    assert payload["answers"][3]["extra_input"] == "备注"
    assert "tags" not in payload["answers"][0]
    query = collection.find_one.await_args.args[0]
    assert query == {
        "uid": "user-001",
        "_id": ObjectId("69b10cc516d7472aedf6bb80"),
    }


@pytest.mark.asyncio
async def test_somni_quiz_service_resolves_option_codes_to_labels() -> None:
    age_id = "a0f15f100000000000000001"
    habit_id = "a0f15f100000000000000010"
    doc = {
        "_id": ObjectId(),
        "uid": "user-001",
        "answers": [
            {
                "question_id": age_id,
                "input_type": "radio",
                "title": "您的年龄段是？",
                "value": "A",
                "extra_input": "",
            },
            {
                "question_id": habit_id,
                "input_type": "checkbox",
                "title": "睡眠困扰",
                "value": ["B", "H"],
                "extra_input": "自定义说明",
            },
            {
                "question_id": "a0f15f100000000000000002",
                "input_type": "time_picker",
                "title": "作息",
                "value": [{"value": "23:30", "label": "上床时间"}],
            },
            {
                "question_id": "a0f15f100000000000000029",
                "input_type": "composite",
                "title": "不被打扰的时间",
                "value": [
                    {"key": "morning", "title": "清晨起床后", "input_type": "slider", "value": 30},
                    {
                        "key": "sound",
                        "title": "声音",
                        "input_type": "checkbox",
                        "value": ["A"],
                    },
                ],
            },
        ],
    }
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=doc)
    questions = [
        {
            "_id": ObjectId(age_id),
            "input_type": "radio",
            "options": {
                "label": "",
                "items": [
                    {"key": "opt_A", "value": "A", "label": "18—24岁"},
                    {"key": "opt_B", "value": "B", "label": "25—34岁"},
                ],
            },
        },
        {
            "_id": ObjectId(habit_id),
            "input_type": "checkbox",
            "options": {
                "items": [
                    {"key": "opt_B", "value": "B", "label": "越想赶紧睡着反而越紧张"},
                    {"key": "opt_H", "value": "H", "label": "", "is_input": True},
                ]
            },
        },
        {
            "_id": ObjectId("a0f15f100000000000000029"),
            "input_type": "composite",
            "options": [
                {"key": "morning", "title": "清晨起床后", "input_type": "slider"},
                {
                    "key": "sound",
                    "title": "声音",
                    "input_type": "checkbox",
                    "options": {"items": [{"key": "opt_A", "value": "A", "label": "自然声"}]},
                },
            ],
        },
    ]
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=_db_collections(collection, questions))
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)
    svc = QuizService(client, Settings())
    payload = await svc.get_answer("user-001", str(doc["_id"]))
    assert payload["answers"][0]["value"] == "18—24岁"
    assert payload["answers"][1]["value"] == ["越想赶紧睡着反而越紧张", "自定义说明"]
    assert payload["answers"][2]["value"] == [{"value": "23:30", "label": "上床时间"}]
    assert payload["answers"][3]["value"][0]["value"] == 30
    assert payload["answers"][3]["value"][1]["value"] == ["自然声"]


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
async def test_somni_quiz_service_loads_latest_when_answer_id_empty() -> None:
    doc = {
        "_id": ObjectId(),
        "uid": "user-001",
        "create_time": "2026-09-02T10:00:00Z",
        "answers": [
            {
                "question_id": "q1",
                "input_type": "radio",
                "title": "您的性别是？",
                "value": {"option_id": "A", "option_text": "先生"},
            }
        ],
    }
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=[doc])
    collection = MagicMock()
    collection.find = MagicMock(return_value=cursor)
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=_db_collections(collection))
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)

    svc = QuizService(client, Settings(somni_mongo_answers_collection="somni_quiz_answers"))
    payload = await svc.get_answer("user-001", "")

    collection.find.assert_called_once_with({"uid": "user-001"})
    cursor.sort.assert_called_once_with("create_time", -1)
    cursor.limit.assert_called_once_with(1)
    assert payload["answers"][0]["question_id"] == "q1"
    assert payload["answers"][0]["value"] == "先生"


@pytest.mark.asyncio
async def test_somni_quiz_service_latest_not_found() -> None:
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=[])
    collection = MagicMock()
    collection.find = MagicMock(return_value=cursor)
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)

    svc = QuizService(client, Settings(somni_mongo_answers_collection="somni_quiz_answers"))
    with pytest.raises(AppError) as exc:
        await svc.get_answer("u1", "")
    assert exc.value.status_code == 404
    assert "暂无答卷" in exc.value.message


@pytest.mark.asyncio
async def test_somni_quiz_service_requires_mongo() -> None:
    svc = QuizService(None, Settings())
    with pytest.raises(MongoNotConfiguredError):
        await svc.get_answer("u1", "a1")
