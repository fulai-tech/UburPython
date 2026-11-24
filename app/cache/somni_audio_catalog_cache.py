"""量产 GetAudio 原料目录 Redis 缓存。只存标签/vip 等骨架，不含展示文案。"""

from __future__ import annotations

import json
import math
from typing import Any

from loguru import logger

from app.cache.somni_audio_locale_cache import (
    LOCALE_KEY_ROOT,
    clear_somni_audio_locale_cache,
)
from app.core.bson_util import bson_to_jsonable

# 固定键名；结构变更靠重建脚本 / warm 覆盖，并 scan 清掉历史杂键。
CATALOG_KEY_ROOT = "somni:audio:catalog"
CATALOG_MATERIALS_TOKEN = "materials"


def catalog_cache_key(*, from_es: bool, language: str = CATALOG_MATERIALS_TOKEN) -> str:
    """language 参数保留兼容旧调用；原料目录固定 materials 段。"""
    del language
    source = "es" if from_es else "mongo"
    return f"{CATALOG_KEY_ROOT}:{source}:{CATALOG_MATERIALS_TOKEN}"


class SomniAudioCatalogCache:
    def __init__(self, redis: Any, *, ttl_sec: float) -> None:
        self._redis = redis
        self._ttl_sec = ttl_sec

    async def get(self, *, from_es: bool, language: str = CATALOG_MATERIALS_TOKEN) -> list[dict[str, Any]] | None:
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
        language: str = CATALOG_MATERIALS_TOKEN,
        docs: list[dict[str, Any]],
    ) -> bool:
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
            return False
        return True


async def clear_somni_audio_catalog_cache(
    redis: Any,
    *,
    keep: str | set[str] | None = None,
) -> int:
    """删除原料目录缓存（及同次失效的语言表缓存）。keep 指定的当前键保留。热榜不动。"""
    keep_set: set[str] = set()
    if isinstance(keep, str) and keep:
        keep_set.add(keep)
    elif isinstance(keep, set):
        keep_set = {key for key in keep if key}

    keys: list[str] = []
    async for key in redis.scan_iter(match=f"{CATALOG_KEY_ROOT}:*", count=100):
        text = str(key)
        if text in keep_set:
            continue
        keys.append(text)
    deleted = 0
    if keys:
        deleted = int(await redis.delete(*keys) or 0)
    # 写入失效时一并清语言表；warm 会传入 keep 集合保留刚写入的 locale 键。
    locale_keep = {key for key in keep_set if key.startswith(f"{LOCALE_KEY_ROOT}:")}
    deleted += await clear_somni_audio_locale_cache(redis, keep=locale_keep or None)
    return deleted
