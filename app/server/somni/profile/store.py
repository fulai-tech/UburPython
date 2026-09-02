"""量产用户画像 Mongo 查询。"""

from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings
from app.core.exceptions import MongoNotConfiguredError


class ProfileStore:
    def __init__(self, client: AsyncIOMotorClient | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def _db(self) -> AsyncIOMotorDatabase:
        if self._client is None:
            raise MongoNotConfiguredError()
        return self._client[self._settings.somni_mongo_db]

    async def find_long_term_profile(self, uid: str) -> dict[str, Any] | None:
        cursor = (
            self._db()[self._settings.somni_mongo_user_profile_long_terms_collection]
            .find({"uid": uid, "status": "valid"})
            .sort("effective_from", -1)
            .limit(1)
        )
        docs = await cursor.to_list(length=1)
        return docs[0] if docs else None

    async def find_short_term_profile(
        self, uid: str, record_date: str
    ) -> dict[str, Any] | None:
        cursor = (
            self._db()[self._settings.somni_mongo_user_profiles_short_terms_collection]
            .find({"uid": uid, "record_date": record_date, "status": "active"})
            .sort("snapshot_revision", -1)
            .limit(1)
        )
        docs = await cursor.to_list(length=1)
        return docs[0] if docs else None
