"""同一条原料的 Mongo 租约。过期判断不能在 Python 里比较两种 datetime。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import asyncio
from types import SimpleNamespace

import pytest
from pymongo.errors import DuplicateKeyError

from app.server.somni.audio import index_queue


class _Leases:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    async def insert_one(self, doc: dict) -> None:
        if doc["_id"] in self.docs:
            raise DuplicateKeyError("dup")
        self.docs[doc["_id"]] = dict(doc)

    async def find_one_and_delete(self, query: dict) -> dict | None:
        current = self.docs.get(query["_id"])
        if current is None:
            return None
        expires_at = current["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= query["expires_at"]["$lte"]:
            return self.docs.pop(query["_id"])
        return None

    async def delete_one(self, query: dict) -> None:
        current = self.docs.get(query["_id"])
        if current is not None and current.get("token") == query.get("token"):
            self.docs.pop(query["_id"], None)

    async def update_one(self, query: dict, update: dict) -> SimpleNamespace:
        current = self.docs.get(query["_id"])
        if current is None or current.get("token") != query.get("token"):
            return SimpleNamespace(matched_count=0)
        current["expires_at"] = update["$set"]["expires_at"]
        return SimpleNamespace(matched_count=1)


@pytest.mark.asyncio
async def test_expired_naive_lease_can_be_taken_over() -> None:
    leases = _Leases()
    expired = (datetime.now(UTC) - timedelta(seconds=5)).replace(tzinfo=None)
    leases.docs["mat"] = {"_id": "mat", "token": "old", "expires_at": expired}

    await index_queue._acquire_lease(leases, "mat", "new")

    assert leases.docs["mat"]["token"] == "new"


@pytest.mark.asyncio
async def test_active_lease_is_left_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_queue, "_WAIT_SECONDS", 0.0)
    leases = _Leases()
    expires_at = (datetime.now(UTC) + timedelta(seconds=30)).replace(tzinfo=None)
    leases.docs["mat"] = {"_id": "mat", "token": "holder", "expires_at": expires_at}

    with pytest.raises(RuntimeError, match="仍在进行"):
        await index_queue._acquire_lease(leases, "mat", "waiter")

    assert leases.docs["mat"]["token"] == "holder"


def _lease_service(leases: _Leases) -> SimpleNamespace:
    return SimpleNamespace(_db=lambda: {index_queue._COLLECTION: leases})


@pytest.mark.asyncio
async def test_lease_is_renewed_until_the_write_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_queue, "_RENEW_SECONDS", 0.02)
    leases = _Leases()

    async with index_queue.material_index_turn(_lease_service(leases), "mat"):
        first = leases.docs["mat"]["expires_at"]
        await asyncio.sleep(0.05)
        assert leases.docs["mat"]["expires_at"] > first

    assert "mat" not in leases.docs


@pytest.mark.asyncio
async def test_lost_lease_cancels_the_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_queue, "_RENEW_SECONDS", 0.02)
    leases = _Leases()

    with pytest.raises(asyncio.CancelledError):
        async with index_queue.material_index_turn(_lease_service(leases), "mat"):
            leases.docs["mat"]["token"] = "taken"
            await asyncio.sleep(0.08)

    assert leases.docs["mat"]["token"] == "taken"
