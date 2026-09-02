"""ProfileStore 查询行为。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.server.somni.profile.store import ProfileStore


def _cursor(*, docs: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(return_value=docs)
    return cursor


@pytest.mark.asyncio
async def test_find_long_term_profile_latest_valid() -> None:
    collection = MagicMock()
    collection.find.return_value = _cursor(
        docs=[{"uid": "u1", "status": "valid", "long_term_profile": {}}]
    )
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ProfileStore(client, Settings())

    doc = await store.find_long_term_profile("u1")

    assert doc is not None
    collection.find.assert_called_once_with({"uid": "u1", "status": "valid"})
    collection.find.return_value.sort.assert_called_once_with("effective_from", -1)
    db.__getitem__.assert_called_with("somni_user_profile_long_terms")


@pytest.mark.asyncio
async def test_find_long_term_profile_none_when_missing() -> None:
    collection = MagicMock()
    collection.find.return_value = _cursor(docs=[])
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ProfileStore(client, Settings())
    assert await store.find_long_term_profile("u1") is None


@pytest.mark.asyncio
async def test_find_short_term_profile_active_revision() -> None:
    collection = MagicMock()
    collection.find.return_value = _cursor(
        docs=[{"uid": "u1", "record_date": "2026-09-01", "status": "active"}]
    )
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = db
    store = ProfileStore(client, Settings())

    doc = await store.find_short_term_profile("u1", "2026-09-01")

    assert doc is not None
    collection.find.assert_called_once_with(
        {"uid": "u1", "record_date": "2026-09-01", "status": "active"}
    )
    collection.find.return_value.sort.assert_called_once_with("snapshot_revision", -1)
    db.__getitem__.assert_called_with("somni_user_profiles_short_terms")
