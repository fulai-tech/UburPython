"""量产音频语言表 Redis 缓存。按 language 存 material_id → 展示字段。"""

from __future__ import annotations

import json
import math
from typing import Any

from loguru import logger

from app.core.bson_util import bson_to_jsonable

# 固定键名；按语言一份 map。结构变更靠重建脚本 / warm 覆盖。
LOCALE_KEY_ROOT = "somni:audio:locale"


def locale_cache_key(language: str) -> str:
    lang = (language or "").strip() or "zh"
    return f"{LOCALE_KEY_ROOT}:{lang}"


def normalize_locale_display(row: dict[str, Any]) -> dict[str, str]:
    """语言表缓存/出参用的展示字段；缺失为空串。"""
    return {
        "audio_name": str(row.get("audio_name") or ""),
        "summary": str(row.get("summary") or row.get("short_description") or ""),
        "description": str(row.get("description") or ""),
        "audio_url": str(row.get("audio_url") or "").strip(),
        "cover_url": str(row.get("cover_url") or "").strip(),
    }


class SomniAudioLocaleCache:
    def __init__(self, redis: Any, *, ttl_sec: float) -> None:
        self._redis = redis
        self._ttl_sec = ttl_sec

    async def get(self, language: str) -> dict[str, dict[str, str]] | None:
        key = locale_cache_key(language)
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            logger.warning("读取量产音频语言表缓存失败：{}", exc)
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("量产音频语言表缓存不是合法 JSON，key={}", key)
            return None
        if not isinstance(payload, dict):
            return None
        out: dict[str, dict[str, str]] = {}
        for material_id, row in payload.items():
            if not isinstance(row, dict):
                continue
            mid = str(material_id or "").strip()
            if mid:
                out[mid] = normalize_locale_display(row)
        return out

    async def set(self, language: str, by_material_id: dict[str, dict[str, str]]) -> bool:
        key = locale_cache_key(language)
        payload = json.dumps(
            bson_to_jsonable(by_material_id),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            if self._ttl_sec <= 0:
                await self._redis.set(key, payload)
            else:
                await self._redis.set(key, payload, ex=max(1, math.ceil(self._ttl_sec)))
        except Exception as exc:
            logger.warning("写入量产音频语言表缓存失败：{}", exc)
            return False
        return True


async def clear_somni_audio_locale_cache(
    redis: Any,
    *,
    keep: str | set[str] | None = None,
) -> int:
    """删除语言表缓存。keep 中的键保留。"""
    keep_set: set[str] = set()
    if isinstance(keep, str) and keep:
        keep_set.add(keep)
    elif isinstance(keep, set):
        keep_set = {key for key in keep if key}

    keys: list[str] = []
    async for key in redis.scan_iter(match=f"{LOCALE_KEY_ROOT}:*", count=100):
        text = str(key)
        if text in keep_set:
            continue
        keys.append(text)
    if not keys:
        return 0
    deleted = await redis.delete(*keys)
    return int(deleted or 0)
