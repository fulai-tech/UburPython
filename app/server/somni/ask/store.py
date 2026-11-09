"""AskContext 只读 Mongo 访问。这条链路只调用 find。"""

from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings
from app.core.exceptions import MongoNotConfiguredError


class AskContextStore:
    def __init__(self, client: AsyncIOMotorClient | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def _db(self) -> AsyncIOMotorDatabase:
        if self._client is None:
            raise MongoNotConfiguredError()
        return self._client[self._settings.somni_mongo_db]

    def _physical(self, logical: str) -> str:
        names = {
            "somni_sleep_reports": self._settings.somni_mongo_sleep_reports_collection,
            "somni_plans": self._settings.somni_mongo_plans_collection,
            "somni_acoustic_clips": self._settings.somni_mongo_acoustic_clips_collection,
            "somni_wearable_daily_profiles": self._settings.somni_mongo_wearable_daily_collection,
        }
        return names[logical]

    async def find(self, logical: str, mongo_filter: dict[str, Any]) -> list[dict[str, Any]]:
        cursor = self._db()[self._physical(logical)].find(mongo_filter)
        return await cursor.to_list(length=None)

    async def find_chat_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        cursor = (
            self._db()[self._settings.somni_mongo_chat_messages_collection]
            .find({"conversation_id": conversation_id})
            .sort("message_index", 1)
        )
        return await cursor.to_list(length=None)
