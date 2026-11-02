"""量产 GetAudio 目录 Redis 缓存。按语言、数据源各存一份已映射文档。"""

from __future__ import annotations

import json
import math
from typing import Any

from loguru import logger

from app.core.audio_tags import audio_tag_contract_version
from app.core.bson_util import bson_to_jsonable

# 目录缓存都挂在这个前缀下。写入时覆盖当前键，并删掉其它版本，避免旧键占着内存。
CATALOG_KEY_ROOT = "somni:audio:catalog"
CATALOG_KEY_PREFIX = f"{CATALOG_KEY_ROOT}:v1"


def catalog_cache_key(*, from_es: bool, language: str) -> str:
    source = "es" if from_es else "mongo"
    return (
        f"{CATALOG_KEY_PREFIX}:{audio_tag_contract_version()}:{source}:{language.strip() or 'zh'}"
    )


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


async def clear_somni_audio_catalog_cache(redis: Any, *, keep: str | None = None) -> int:
    """删除目录缓存。keep 指定的当前键保留，热榜等其它前缀不动。"""
    keys: list[str] = []
    async for key in redis.scan_iter(match=f"{CATALOG_KEY_ROOT}:*", count=100):
        text = str(key)
        if keep is not None and text == keep:
            continue
        keys.append(text)
    if not keys:
        return 0
    deleted = await redis.delete(*keys)
    return int(deleted or 0)
