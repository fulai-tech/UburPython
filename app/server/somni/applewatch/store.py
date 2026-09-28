"""量产 Apple Watch 聚合 Mongo 查询。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings
from app.core.exceptions import MongoNotConfiguredError


class AppleWatchStore:
    def __init__(self, client: AsyncIOMotorClient | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def _db(self) -> AsyncIOMotorDatabase:
        if self._client is None:
            raise MongoNotConfiguredError()
        return self._client[self._settings.somni_mongo_db]

    async def list_step_points(
        self,
        *,
        uid: str,
        timezone: str,
        start_at: datetime,
        end_at: datetime,
    ) -> list[dict[str, Any]]:
        cursor = (
            self._db()[self._settings.somni_mongo_wearable_raw_points_collection]
            .find(
                {
                    "original_type": "STEPS",
                    "status": 1,
                    "user_id": uid,
                    "timezone": timezone,
                    "start_at": {"$gte": start_at, "$lt": end_at},
                }
            )
            .sort("start_at", 1)
        )
        return await cursor.to_list(length=50_000)

    async def find_alarm(
        self,
        *,
        uid: str,
        device_id: str,
        week_day: int,
    ) -> dict[str, Any] | None:
        return await self._db()[self._settings.somni_mongo_alarms_collection].find_one(
            {
                "uid": uid,
                "device_id": device_id,
                "week_day": week_day,
            }
        )
