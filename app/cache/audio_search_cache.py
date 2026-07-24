"""音频检索结果 Redis 缓存（LRU + 固定 TTL）。"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol

from loguru import logger

from app.core.config import Settings
from app.schemas.audio import SearchAudioRequest

AUDIO_SEARCH_KEY_PREFIX = "audio_search_v2+"
AUDIO_SEARCH_LRU_INDEX_KEY = "audio_search:_lru_index"
_LRU_INDEX_KEY = AUDIO_SEARCH_LRU_INDEX_KEY  # 兼容旧测试名
_WEEK_SECONDS = 7 * 24 * 60 * 60


class RedisLike(Protocol):
    async def get(self, key: str | bytes) -> bytes | None: ...

    async def set(self, key: str | bytes, value: str | bytes, ex: int | None = None) -> Any: ...

    async def delete(self, *keys: str | bytes) -> Any: ...

    async def zadd(self, name: str | bytes, mapping: dict[Any, float]) -> Any: ...

    async def zcard(self, name: str | bytes) -> int: ...

    async def zrange(self, name: str | bytes, start: int, end: int) -> list[bytes]: ...

    async def zrem(self, name: str | bytes, *members: str | bytes) -> Any: ...

    async def close(self) -> None: ...


def build_audio_search_cache_key(query: SearchAudioRequest) -> str:
    """key = sha256(audio_search_v2+{稳定序列化单条 query})。"""
    body = json.dumps(
        query.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{AUDIO_SEARCH_KEY_PREFIX}{body}".encode()).hexdigest()
    return digest


class AudioSearchCache:
    """检索 materials 缓存：命中不续 TTL、超量 LRU 淘汰、过期清幽灵、CUD 全清。"""

    def __init__(
        self,
        redis: RedisLike,
        *,
        max_size: int = 2048,
        ttl_sec: int = _WEEK_SECONDS,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if ttl_sec < 1:
            raise ValueError("ttl_sec must be >= 1")
        self._redis = redis
        self._max_size = max_size
        self._ttl_sec = ttl_sec

    @property
    def redis(self) -> RedisLike:
        return self._redis

    async def get(self, query: SearchAudioRequest) -> list[dict[str, Any]] | None:
        key = build_audio_search_cache_key(query)
        raw = await self._redis.get(key)
        if raw is None:
            await self._redis.zrem(_LRU_INDEX_KEY, key)
            logger.info("检索缓存未命中，key={}", key[:12])
            return None
        await self._bump_lru(key)
        logger.info("检索缓存命中，key={}", key[:12])
        return json.loads(raw)

    async def set(self, query: SearchAudioRequest, materials: list[dict[str, Any]]) -> None:
        key = build_audio_search_cache_key(query)
        payload = json.dumps(materials, ensure_ascii=False, separators=(",", ":"))
        await self._redis.set(key, payload, ex=self._ttl_sec)
        await self._redis.zadd(_LRU_INDEX_KEY, {key: time.time()})
        await self._prune_ghosts()
        await self._evict_if_needed()
        logger.info("检索缓存已写入，key={} materials={}", key[:12], len(materials))

    async def clear_all(self) -> None:
        keys = await self._redis.zrange(_LRU_INDEX_KEY, 0, -1)
        if keys:
            await self._redis.delete(*keys)
        await self._redis.delete(_LRU_INDEX_KEY)
        logger.info("已清除全部音频检索缓存，数量={}", len(keys))

    async def close(self) -> None:
        aclose = getattr(self._redis, "aclose", None)
        if aclose is not None:
            await aclose()
            return
        close = getattr(self._redis, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result

    async def _bump_lru(self, key: str) -> None:
        """仅更新 LRU 分数；不重置结果 key 的 TTL。"""
        await self._redis.zadd(_LRU_INDEX_KEY, {key: time.time()})

    async def _prune_ghosts(self) -> None:
        """索引里有、结果 key 已不存在（含 TTL 过期）→ 从 LRU 去掉。"""
        members = await self._redis.zrange(_LRU_INDEX_KEY, 0, -1)
        ghosts: list[str | bytes] = []
        for member in members:
            raw = await self._redis.get(member)
            if raw is None:
                ghosts.append(member)
        if ghosts:
            await self._redis.zrem(_LRU_INDEX_KEY, *ghosts)

    async def _evict_if_needed(self) -> None:
        size = await self._redis.zcard(_LRU_INDEX_KEY)
        overflow = size - self._max_size
        if overflow <= 0:
            return
        victims = await self._redis.zrange(_LRU_INDEX_KEY, 0, overflow - 1)
        if not victims:
            return
        await self._redis.delete(*victims)
        await self._redis.zrem(_LRU_INDEX_KEY, *victims)


async def create_audio_search_cache(settings: Settings) -> AudioSearchCache | None:
    """按 REDIS_URL 建缓存；空 URL 或 debug 下连不上则返回 None。"""
    if not settings.redis_url.strip():
        logger.warning("未配置 REDIS_URL，音频检索缓存已关闭")
        return None
    from redis.asyncio import Redis

    max_connections = max(1, settings.redis_max_connections)
    client = Redis.from_url(
        settings.redis_url,
        decode_responses=False,
        max_connections=max_connections,
    )
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        if settings.app_debug:
            logger.warning("Redis 不可用（调试模式，跳过检索缓存）：{}", exc)
            return None
        raise
    logger.info(
        "已连接 Redis 检索缓存，max_size={} ttl_sec={} max_connections={}",
        settings.search_cache_max_size,
        settings.search_cache_ttl_sec,
        max_connections,
    )
    return AudioSearchCache(
        client,
        max_size=settings.search_cache_max_size,
        ttl_sec=settings.search_cache_ttl_sec,
    )


async def shutdown_audio_search_cache(cache: AudioSearchCache | None) -> None:
    """进程退出前清空检索缓存并关闭连接。"""
    if cache is None:
        return
    try:
        await cache.clear_all()
    except Exception as exc:
        logger.error("关闭前清除检索缓存失败：{}", exc)
    await cache.close()
