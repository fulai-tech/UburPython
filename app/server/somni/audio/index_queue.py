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
from pymongo import ReturnDocument
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


def fence_filter(fence: int | None) -> dict[str, Any]:
    """写入条件：原料文档还没被更高编号的租约写过。字段缺失（老数据）也允许写。"""
    if fence is None:
        return {}
    return {"$or": [{"fence_token": None}, {"fence_token": {"$lte": fence}}]}


def fence_set(fence: int | None) -> dict[str, Any]:
    """写入成功后把本次租约编号落到文档上，供后来者和补偿比对。"""
    return {} if fence is None else {"fence_token": fence}


async def _acquire_lease(collection: Any, material_id: str, token: str) -> int:
    """取得租约，返回本次编号。编号只增不减：接管时原地 +1，释放时不删除文档。"""
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while True:
        now = utc_now()
        try:
            await collection.insert_one(
                {"_id": material_id, "token": token, "expires_at": now + _LEASE, "fence_token": 1}
            )
            return 1
        except DuplicateKeyError:
            # 过期与否交给 Mongo 比较。读回的时间不带时区，和 utc_now() 不能直接比。
            # 原地接管而不是删除再插入，否则计数器会丢，编号会倒退。
            stolen = await collection.find_one_and_update(
                {"_id": material_id, "expires_at": {"$lte": now}},
                {
                    "$set": {"token": token, "expires_at": now + _LEASE},
                    "$inc": {"fence_token": 1},
                },
                return_document=ReturnDocument.AFTER,
            )
            if stolen is not None:
                return int(stolen["fence_token"])
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("同一条原料的索引写入仍在进行，请重试")
            await asyncio.sleep(_POLL_SECONDS)


async def _release_lease(collection: Any, material_id: str, token: str) -> None:
    """释放时只把租约标记为过期，保留文档和编号计数器。"""
    try:
        await collection.update_one(
            {"_id": material_id, "token": token},
            {"$set": {"expires_at": utc_now()}},
        )
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
async def material_index_turn(service: Any, material_id: str) -> AsyncIterator[int | None]:
    """同一 material_id 同时只有一个写入。yield 本次租约编号（无 Mongo 时为 None）。

    调用方在退出前写完 Mongo 和 ES，写入时带上编号，旧编号的写入会被拒绝。
    """
    local = await _process_lock(material_id)
    async with local:
        collection = _lease_collection(service)
        token = uuid.uuid4().hex
        stop = asyncio.Event()
        renew: asyncio.Task[None] | None = None
        fence: int | None = None
        holder = asyncio.current_task()
        if holder is not None:
            _LEASE_HOLDERS[material_id] = holder
        if collection is not None:
            fence = await _acquire_lease(collection, material_id, token)
            renew = asyncio.create_task(
                _keep_lease_alive(collection, material_id, token, stop)
            )
        try:
            yield fence
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
