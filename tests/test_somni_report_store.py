"""ReportStore 查询行为。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.server.somni.report.store import ReportStore


@pytest.mark.asyncio
async def test_list_events_reads_sleep_events_field() -> None:
    collection = MagicMock()
    collection.find_one = AsyncMock(
        return_value={
            "uid": "u1",
            "record_date": "2026-08-10",
            "sleep_events": [
                {"type": "abnormal", "_id": "a1"},
                {"type": "intervention", "related_event_id": "a1"},
            ],
        }
    )
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ReportStore(client, Settings())
    events = await store.list_events("u1", "2026-08-10")
    assert len(events) == 2
    assert events[0]["type"] == "abnormal"


@pytest.mark.asyncio
async def test_list_events_empty_when_missing() -> None:
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ReportStore(client, Settings())
    assert await store.list_events("u1", "2026-08-10") == []


@pytest.mark.asyncio
async def test_find_user_profile_by_uid_and_record_date() -> None:
    collection = MagicMock()
    collection.find_one = AsyncMock(
        return_value={
            "uid": "u1",
            "record_date": "2026-08-30",
            "profile_text": "画像摘要",
        }
    )
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ReportStore(client, Settings())
    doc = await store.find_user_profile("u1", "2026-08-30")
    assert doc is not None
    assert doc["profile_text"] == "画像摘要"
    collection.find_one.assert_awaited_once_with(
        {"uid": "u1", "record_date": "2026-08-30"}
    )
    db.__getitem__.assert_called_with("somni_user_profiles")


@pytest.mark.asyncio
async def test_find_user_profile_none_when_missing() -> None:
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ReportStore(client, Settings())
    assert await store.find_user_profile("u1", "2026-08-30") is None


@pytest.mark.asyncio
async def test_list_records_by_date_range() -> None:
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(
        return_value=[{"uid": "u1", "record_date": "2026-09-22"}]
    )
    collection = MagicMock()
    collection.find = MagicMock(return_value=cursor)
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ReportStore(client, Settings())
    docs = await store.list_records("u1", "2026-09-15", "2026-09-28")
    assert len(docs) == 1
    collection.find.assert_called_once_with(
        {"uid": "u1", "record_date": {"$gte": "2026-09-15", "$lte": "2026-09-28"}}
    )
    cursor.sort.assert_called_once_with("record_date", 1)


@pytest.mark.asyncio
async def test_find_user_timezone() -> None:
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value={"uid": "u1", "timezone": "UTC"})
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ReportStore(client, Settings())
    assert await store.find_user_timezone("u1") == "UTC"
    collection.find_one.assert_awaited_once_with({"uid": "u1"}, {"timezone": 1})
    db.__getitem__.assert_called_with("somni_users")
