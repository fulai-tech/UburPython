"""量产报告 Mongo 查询。"""

from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings
from app.core.exceptions import MongoNotConfiguredError
from app.server.somni.report.calc import local_day_utc_range


class ReportStore:
    def __init__(self, client: AsyncIOMotorClient | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def _db(self) -> AsyncIOMotorDatabase:
        if self._client is None:
            raise MongoNotConfiguredError()
        return self._client[self._settings.somni_mongo_db]

    async def find_device_id(self, uid: str) -> str | None:
        doc = await self._db()[self._settings.somni_mongo_devices_collection].find_one(
            {"bind_uid": uid},
            {"device_id": 1},
        )
        if not doc:
            return None
        device_id = doc.get("device_id")
        return str(device_id) if device_id else None

    async def list_telemetry(
        self,
        device_id: str,
        metric: str,
        record_date: str,
    ) -> list[dict[str, Any]]:
        start, end = local_day_utc_range(record_date)
        cursor = self._db()[self._settings.somni_mongo_telemetry_collection].find(
            {
                "device_id": device_id,
                "metric": metric,
                "ts": {"$gte": start, "$lt": end},
            }
        ).sort("ts", 1)
        return await cursor.to_list(length=10_000)

    async def find_record(self, uid: str, record_date: str) -> dict[str, Any] | None:
        return await self._db()[self._settings.somni_mongo_records_collection].find_one(
            {"uid": uid, "record_date": record_date}
        )

    async def find_sleep_report(
        self, uid: str, record_date: str
    ) -> dict[str, Any] | None:
        return await self._db()[
            self._settings.somni_mongo_sleep_reports_collection
        ].find_one({"uid": uid, "record_date": record_date})

    async def list_events(self, uid: str, record_date: str) -> list[dict[str, Any]]:
        doc = await self._db()[self._settings.somni_mongo_events_collection].find_one(
            {"uid": uid, "record_date": record_date}
        )
        if not doc:
            return []
        nested = doc.get("sleep_events")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
        return [doc]
