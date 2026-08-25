"""量产问卷：按 uid + answer_id 查 somni_quiz_answers。"""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.bson_util import bson_to_jsonable
from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError, MongoNotConfiguredError


class QuizService:
    def __init__(self, client: AsyncIOMotorClient | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def get_answer(self, uid: str, answer_id: str) -> dict[str, Any]:
        if self._client is None:
            raise MongoNotConfiguredError()
        collection = self._client[self._settings.somni_mongo_db][
            self._settings.somni_mongo_answers_collection
        ]
        doc = await collection.find_one({"uid": uid, **_id_query(answer_id)})
        if doc is None:
            raise AppError(
                message=f"答卷不存在：{answer_id}",
                status_code=HttpStatus.NOT_FOUND,
            )
        raw = bson_to_jsonable(doc)
        answers = [_normalize_answer(item) for item in (raw.get("answers") or [])]
        return {"answers": answers}


def _id_query(answer_id: str) -> dict[str, Any]:
    try:
        return {"_id": ObjectId(answer_id)}
    except InvalidId:
        return {"$or": [{"_id": answer_id}, {"id": answer_id}]}


def _normalize_answer(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise AppError(
            message="答卷明细格式非法",
            status_code=HttpStatus.INTERNAL_SERVER_ERROR,
        )
    return {
        "question_id": str(item.get("question_id") or ""),
        "input_type": str(item.get("input_type") or ""),
        "title": str(item.get("title") or ""),
        "value": item.get("value"),
        "extra_input": str(item.get("extra_input") or ""),
    }
