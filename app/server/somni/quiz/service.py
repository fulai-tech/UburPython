"""量产问卷：按 uid + answer_id 查 somni_quiz_answers；无 answer_id 取最新。"""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.bson_util import bson_to_jsonable
from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError, MongoNotConfiguredError
from app.server.somni.quiz.answer_value import concrete_answer_value

_CHOICE_TYPES = {"radio", "select", "checkbox"}


class QuizService:
    def __init__(self, client: AsyncIOMotorClient | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def get_answer(self, uid: str, answer_id: str) -> dict[str, Any]:
        if self._client is None:
            raise MongoNotConfiguredError()
        db = self._client[self._settings.somni_mongo_db]
        collection = db[self._settings.somni_mongo_answers_collection]
        if answer_id.strip():
            doc = await collection.find_one({"uid": uid, **_id_query(answer_id)})
            missing_msg = f"答卷不存在：{answer_id}"
        else:
            cursor = collection.find({"uid": uid}).sort("create_time", -1).limit(1)
            docs = await cursor.to_list(length=1)
            doc = docs[0] if docs else None
            missing_msg = "该用户暂无答卷"
        if doc is None:
            raise AppError(message=missing_msg, status_code=HttpStatus.NOT_FOUND)
        raw = bson_to_jsonable(doc)
        answers = raw.get("answers") or []
        question_ids = [
            str(item.get("question_id") or "")
            for item in answers
            if isinstance(item, dict) and item.get("question_id")
        ]
        questions = await self._load_questions(db, question_ids)
        return {
            "answers": [
                _normalize_answer(item, questions.get(str(item.get("question_id") or "")))
                for item in answers
            ]
        }

    async def _load_questions(self, db: Any, question_ids: list[str]) -> dict[str, dict]:
        if not question_ids:
            return {}
        oids = [ObjectId(qid) for qid in question_ids if ObjectId.is_valid(qid)]
        clauses: list[dict] = []
        if oids:
            clauses.append({"_id": {"$in": oids}})
        clauses.append({"_id": {"$in": question_ids}})
        query = clauses[0] if len(clauses) == 1 else {"$or": clauses}
        cursor = db[self._settings.somni_mongo_questions_collection].find(
            query, {"input_type": 1, "options": 1}
        )
        docs = await cursor.to_list(length=len(question_ids) + len(oids))
        return {str(doc.get("_id")): bson_to_jsonable(doc) for doc in docs if doc.get("_id") is not None}


def _id_query(answer_id: str) -> dict[str, Any]:
    try:
        return {"_id": ObjectId(answer_id)}
    except InvalidId:
        return {"$or": [{"_id": answer_id}, {"id": answer_id}]}


def _normalize_answer(item: Any, question: dict | None = None) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise AppError(
            message="答卷明细格式非法",
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
        )
    options = question.get("options") if isinstance(question, dict) else None
    return {
        "question_id": str(item.get("question_id") or ""),
        "input_type": str(item.get("input_type") or ""),
        "title": str(item.get("title") or ""),
        "value": concrete_answer_value(
            str(item.get("input_type") or ""),
            item.get("value"),
            options,
            str(item.get("extra_input") or ""),
        ),
        "extra_input": str(item.get("extra_input") or ""),
    }
