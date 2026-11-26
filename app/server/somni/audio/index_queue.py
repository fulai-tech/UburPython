"""同一条原料的写入排队。

本进程用锁串行。多实例再用 Mongo 租约：前一次未释放时，后一次等待。
租约过期后下一次可以接手，避免进程退出后队列永远停住。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from loguru import logger
from pymongo.errors import DuplicateKeyError

from app.core.bson_util import utc_now

_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()
_COLLECTION = "somni_audio_index_locks"
_LEASE = timedelta(seconds=120)
_WAIT_SECONDS = 60.0
_POLL_SECONDS = 0.05
_RENEW_SECONDS = 30.0


async def _process_lock(material_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        lock = _LOCKS.get(material_id)
        if lock is None:
            lock = asyncio.Lock()
            _LOCKS[material_id] = lock
        return lock


def _lease_collection(service: Any) -> Any | None:
    getter = getattr(service, "_db", None)
    if not callable(getter):
        return None
    try:
        return getter()[_COLLECTION]
    except Exception:
        return None


async def _acquire_lease(collection: Any, material_id: str, token: str) -> None:
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while True:
        now = utc_now()
        try:
            await collection.insert_one(
                {"_id": material_id, "token": token, "expires_at": now + _LEASE}
            )
            return
        except DuplicateKeyError:
            # 过期与否交给 Mongo 比较。读回的时间不带时区，和 utc_now() 不能直接比。
            stolen = await collection.find_one_and_delete(
                {"_id": material_id, "expires_at": {"$lte": now}}
            )
            if stolen is not None:
                continue
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("同一条原料的索引写入仍在进行，请重试")
            await asyncio.sleep(_POLL_SECONDS)


async def _release_lease(collection: Any, material_id: str, token: str) -> None:
    try:
        await collection.delete_one({"_id": material_id, "token": token})
    except Exception as exc:
        logger.error("释放原料索引租约失败，material_id={}，原因：{}", material_id, exc)


async def _renew_lease(collection: Any, material_id: str, token: str) -> bool:
    """把仍属于本次写入的租约往后推。token 已经不是自己的，说明被别人拿走了。"""
    result = await collection.update_one(
        {"_id": material_id, "token": token},
        {"$set": {"expires_at": utc_now() + _LEASE}},
    )
    return bool(getattr(result, "matched_count", 0))


async def _keep_lease_alive(
    collection: Any,
    material_id: str,
    token: str,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=_RENEW_SECONDS)
            return
        except asyncio.TimeoutError:
            pass
        try:
            kept = await _renew_lease(collection, material_id, token)
        except Exception as exc:
            logger.error("续原料索引租约失败，material_id={}，原因：{}", material_id, exc)
            kept = False
        if kept:
            continue
        logger.error("原料索引租约已丢失，中止本次写入，material_id={}", material_id)
        holder = asyncio.current_task()
        # current_task 是续约任务自己。取消持有写入的那个任务。
        parent = _LEASE_HOLDERS.get(material_id)
        if parent is not None and parent is not holder and not parent.done():
            parent.cancel()
        return


_LEASE_HOLDERS: dict[str, asyncio.Task[Any]] = {}


@asynccontextmanager
async def material_index_turn(service: Any, material_id: str) -> AsyncIterator[None]:
    """同一 material_id 同时只有一个写入。调用方在退出前写完 Mongo 和 ES。"""
    local = await _process_lock(material_id)
    async with local:
        collection = _lease_collection(service)
        token = uuid.uuid4().hex
        stop = asyncio.Event()
        renew: asyncio.Task[None] | None = None
        holder = asyncio.current_task()
        if holder is not None:
            _LEASE_HOLDERS[material_id] = holder
        if collection is not None:
            await _acquire_lease(collection, material_id, token)
            renew = asyncio.create_task(
                _keep_lease_alive(collection, material_id, token, stop)
            )
        try:
            yield
        finally:
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
            stop.set()
            if renew is not None:
                renew.cancel()
                try:
                    await asyncio.shield(renew)
                except asyncio.CancelledError:
                    pass
            if _LEASE_HOLDERS.get(material_id) is holder:
                _LEASE_HOLDERS.pop(material_id, None)
            if collection is not None:
                try:
                    await asyncio.shield(_release_lease(collection, material_id, token))
                except asyncio.CancelledError:
                    pass
