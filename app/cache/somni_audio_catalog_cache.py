"""量产 GetAudio 目录 Redis 缓存。按语言、数据源各存一份已映射文档。"""

from __future__ import annotations

import json
import math
from typing import Any

from loguru import logger

from app.core.bson_util import bson_to_jsonable

CATALOG_KEY_PREFIX = "somni:audio:catalog:v1"


def catalog_cache_key(*, from_es: bool, language: str) -> str:
    source = "es" if from_es else "mongo"
    return f"{CATALOG_KEY_PREFIX}:{source}:{language.strip() or 'zh'}"


class SomniAudioCatalogCache:
    def __init__(self, redis: Any, *, ttl_sec: float) -> None:
        self._redis = redis
        self._ttl_sec = ttl_sec

    async def get(self, *, from_es: bool, language: str) -> list[dict[str, Any]] | None:
        key = catalog_cache_key(from_es=from_es, language=language)
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            logger.warning("读取量产 GetAudio 目录缓存失败：{}", exc)
            return None
        if raw is None:
            return None
        try:
            docs = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("量产 GetAudio 目录缓存不是合法 JSON，key={}", key)
            return None
        if not isinstance(docs, list):
            return None
        return docs

    async def set(
        self,
        *,
        from_es: bool,
        language: str,
        docs: list[dict[str, Any]],
    ) -> None:
        key = catalog_cache_key(from_es=from_es, language=language)
        payload = json.dumps(
            bson_to_jsonable(docs),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            if self._ttl_sec <= 0:
                await self._redis.set(key, payload)
            else:
                await self._redis.set(key, payload, ex=max(1, math.ceil(self._ttl_sec)))
        except Exception as exc:
            logger.warning("写入量产 GetAudio 目录缓存失败：{}", exc)


async def clear_somni_audio_catalog_cache(redis: Any) -> int:
    """删除全部语言、Mongo/ES 两路目录缓存。"""
    keys: list[str] = []
    async for key in redis.scan_iter(match=f"{CATALOG_KEY_PREFIX}:*", count=100):
        keys.append(str(key))
    if not keys:
        return 0
    deleted = await redis.delete(*keys)
    return int(deleted or 0)
