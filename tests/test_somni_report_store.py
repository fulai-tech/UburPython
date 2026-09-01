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
