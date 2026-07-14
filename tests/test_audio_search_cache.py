"""AudioSearchCache：key 生成、LRU、TTL 固定、幽灵清理、全量清除。"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

from app.cache.audio_search_cache import (
    AUDIO_SEARCH_KEY_PREFIX,
    AUDIO_SEARCH_LRU_INDEX_KEY,
    AudioSearchCache,
    build_audio_search_cache_key,
)
from app.schemas.audio import SearchAudioRequest


class _FakeRedis:
    """最小异步 Redis 替身，覆盖本缓存用到的命令。"""

    def __init__(self) -> None:
        self.kv: dict[bytes, bytes] = {}
        self.ttl: dict[bytes, float | None] = {}
        self.zsets: dict[bytes, dict[bytes, float]] = {}
        self.now = time.time()

    def _b(self, value: str | bytes) -> bytes:
        return value if isinstance(value, bytes) else value.encode()

    def _purge_expired(self, key: bytes) -> None:
        expire_at = self.ttl.get(key)
        if expire_at is not None and expire_at <= self.now:
            self.kv.pop(key, None)
            self.ttl.pop(key, None)

    async def get(self, key: str | bytes) -> bytes | None:
        k = self._b(key)
        self._purge_expired(k)
        return self.kv.get(k)

    async def set(self, key: str | bytes, value: str | bytes, ex: int | None = None) -> bool:
        k = self._b(key)
        self.kv[k] = self._b(value)
        self.ttl[k] = None if ex is None else self.now + ex
        return True

    async def expire(self, key: str | bytes, seconds: int) -> bool:
        k = self._b(key)
        if k not in self.kv:
            return False
        self.ttl[k] = self.now + seconds
        return True

    async def delete(self, *keys: str | bytes) -> int:
        removed = 0
        for key in keys:
            k = self._b(key)
            if k in self.kv or k in self.zsets:
                removed += 1
            self.kv.pop(k, None)
            self.ttl.pop(k, None)
            self.zsets.pop(k, None)
            for zset in self.zsets.values():
                zset.pop(k, None)
        return removed

    async def zadd(self, name: str | bytes, mapping: dict[str | bytes, float]) -> int:
        zname = self._b(name)
        zset = self.zsets.setdefault(zname, {})
        added = 0
        for member, score in mapping.items():
            m = self._b(member)
            if m not in zset:
                added += 1
            zset[m] = float(score)
        return added

    async def zcard(self, name: str | bytes) -> int:
        return len(self.zsets.get(self._b(name), {}))

    async def zrange(self, name: str | bytes, start: int, end: int) -> list[bytes]:
        zset = self.zsets.get(self._b(name), {})
        ordered = sorted(zset.items(), key=lambda item: item[1])
        members = [member for member, _ in ordered]
        if end == -1:
            end = len(members) - 1
        if end < 0:
            end = len(members) + end
        return members[start : end + 1]

    async def zrem(self, name: str | bytes, *members: str | bytes) -> int:
        zset = self.zsets.get(self._b(name), {})
        removed = 0
        for member in members:
            m = self._b(member)
            if m in zset:
                del zset[m]
                removed += 1
        return removed

    async def close(self) -> None:
        return None


def test_build_cache_key_is_sha_of_prefix_plus_body() -> None:
    request = SearchAudioRequest(query_text="雨声", top_k=5)
    body = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(f"{AUDIO_SEARCH_KEY_PREFIX}{body}".encode()).hexdigest()
    assert build_audio_search_cache_key(request) == expected


@pytest.mark.asyncio
async def test_get_miss_returns_none() -> None:
    cache = AudioSearchCache(_FakeRedis(), max_size=3, ttl_sec=60)
    assert await cache.get(SearchAudioRequest(query_text="x")) is None


@pytest.mark.asyncio
async def test_set_then_get_returns_materials_without_refreshing_ttl() -> None:
    redis = _FakeRedis()
    cache = AudioSearchCache(redis, max_size=3, ttl_sec=100)
    request = SearchAudioRequest(query_text="雨声", top_k=2)
    materials = [{"id": "a1", "audio_name": "雨声"}]

    await cache.set(request, materials)
    key = build_audio_search_cache_key(request)
    expire_at = redis.now + 100
    assert redis.ttl[key.encode()] == pytest.approx(expire_at)

    redis.now += 40
    got = await cache.get(request)
    assert got == materials
    # 命中只更新 LRU，不重置 TTL
    assert redis.ttl[key.encode()] == pytest.approx(expire_at)


@pytest.mark.asyncio
async def test_expired_key_miss_removes_ghost_from_lru_index() -> None:
    redis = _FakeRedis()
    cache = AudioSearchCache(redis, max_size=3, ttl_sec=50)
    request = SearchAudioRequest(query_text="雨声")
    await cache.set(request, [{"id": "a1"}])
    key = build_audio_search_cache_key(request)
    index = redis.zsets[AUDIO_SEARCH_LRU_INDEX_KEY.encode()]
    assert key.encode() in index

    redis.now += 51  # 超过 TTL，kv 过期
    assert await cache.get(request) is None
    assert key.encode() not in redis.zsets.get(AUDIO_SEARCH_LRU_INDEX_KEY.encode(), {})


@pytest.mark.asyncio
async def test_set_prunes_ghosts_before_counting_lru() -> None:
    redis = _FakeRedis()
    cache = AudioSearchCache(redis, max_size=2, ttl_sec=30)

    await cache.set(SearchAudioRequest(query_text="a"), [{"id": "a"}])
    redis.now += 1
    await cache.set(SearchAudioRequest(query_text="b"), [{"id": "b"}])
    # a、b 均过期，留下幽灵索引
    redis.now += 40
    assert await redis.get(build_audio_search_cache_key(SearchAudioRequest(query_text="a"))) is None
    assert await redis.get(build_audio_search_cache_key(SearchAudioRequest(query_text="b"))) is None
    assert await redis.zcard(AUDIO_SEARCH_LRU_INDEX_KEY) == 2

    await cache.set(SearchAudioRequest(query_text="c"), [{"id": "c"}])
    index = redis.zsets.get(AUDIO_SEARCH_LRU_INDEX_KEY.encode(), {})
    assert len(index) == 1
    assert build_audio_search_cache_key(SearchAudioRequest(query_text="c")).encode() in index


@pytest.mark.asyncio
async def test_lru_evicts_oldest_when_over_max_size() -> None:
    redis = _FakeRedis()
    cache = AudioSearchCache(redis, max_size=2, ttl_sec=60)

    await cache.set(SearchAudioRequest(query_text="a"), [{"id": "a"}])
    redis.now += 1
    await cache.set(SearchAudioRequest(query_text="b"), [{"id": "b"}])
    redis.now += 1
    await cache.set(SearchAudioRequest(query_text="c"), [{"id": "c"}])

    assert await cache.get(SearchAudioRequest(query_text="a")) is None
    assert await cache.get(SearchAudioRequest(query_text="b")) == [{"id": "b"}]
    assert await cache.get(SearchAudioRequest(query_text="c")) == [{"id": "c"}]


@pytest.mark.asyncio
async def test_get_touch_updates_lru_order() -> None:
    redis = _FakeRedis()
    cache = AudioSearchCache(redis, max_size=2, ttl_sec=60)

    await cache.set(SearchAudioRequest(query_text="a"), [{"id": "a"}])
    redis.now += 1
    await cache.set(SearchAudioRequest(query_text="b"), [{"id": "b"}])
    redis.now += 1
    assert await cache.get(SearchAudioRequest(query_text="a")) == [{"id": "a"}]
    redis.now += 1
    await cache.set(SearchAudioRequest(query_text="c"), [{"id": "c"}])

    assert await cache.get(SearchAudioRequest(query_text="b")) is None
    assert await cache.get(SearchAudioRequest(query_text="a")) == [{"id": "a"}]
    assert await cache.get(SearchAudioRequest(query_text="c")) == [{"id": "c"}]


@pytest.mark.asyncio
async def test_clear_all_removes_audio_search_entries() -> None:
    cache = AudioSearchCache(_FakeRedis(), max_size=10, ttl_sec=60)
    await cache.set(SearchAudioRequest(query_text="a"), [{"id": "a"}])
    await cache.set(SearchAudioRequest(query_text="b"), [{"id": "b"}])

    await cache.clear_all()

    assert await cache.get(SearchAudioRequest(query_text="a")) is None
    assert await cache.get(SearchAudioRequest(query_text="b")) is None


@pytest.mark.asyncio
async def test_shutdown_cache_clears_all_then_closes() -> None:
    from unittest.mock import AsyncMock, call

    from app.cache.audio_search_cache import shutdown_audio_search_cache

    cache = AsyncMock()
    await shutdown_audio_search_cache(cache)
    assert cache.mock_calls == [call.clear_all(), call.close()]


@pytest.mark.asyncio
async def test_shutdown_cache_none_is_noop() -> None:
    from app.cache.audio_search_cache import shutdown_audio_search_cache

    await shutdown_audio_search_cache(None)


@pytest.mark.asyncio
async def test_shutdown_cache_still_closes_if_clear_fails() -> None:
    from unittest.mock import AsyncMock

    from app.cache.audio_search_cache import shutdown_audio_search_cache

    cache = AsyncMock()
    cache.clear_all = AsyncMock(side_effect=RuntimeError("redis down"))
    await shutdown_audio_search_cache(cache)
    cache.clear_all.assert_awaited_once()
    cache.close.assert_awaited_once()
