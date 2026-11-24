"""量产文本→embedding 向量 Redis 缓存（按语言+类型 Hash，field=sha256）。

结构：
  Redis key: embedding_text:{zh|en}:{query|tag}
  field: sha256(utf-8 原文)   # 不用中文 field
  value: float32 二进制向量

换模型请手动清 embedding_text:* 后重建。
"""

from __future__ import annotations

import array
import hashlib
import struct
import time
from typing import Any, Literal, Protocol

from loguru import logger

from app.es.somni_docs import resolve_search_language

_MAGIC = b"EMB1"
_HEADER = struct.Struct(">4sI")  # magic + dim
_WEEK_SECONDS = 7 * 24 * 60 * 60

EMBEDDING_TEXT_KEY_PREFIX = "embedding_text"
EmbeddingKind = Literal["query", "tag"]


class RedisLike(Protocol):
    async def hget(self, name: str | bytes, key: str | bytes) -> bytes | str | None: ...

    async def hmget(self, name: str | bytes, keys: list[Any]) -> list[bytes | str | None]: ...

    async def hset(self, name: str | bytes, key: str | bytes, value: str | bytes) -> Any: ...

    async def hdel(self, name: str | bytes, *keys: str | bytes) -> Any: ...

    async def delete(self, *keys: str | bytes) -> Any: ...

    async def expire(self, name: str | bytes, time: int) -> Any: ...

    async def zadd(self, name: str | bytes, mapping: dict[Any, float]) -> Any: ...

    async def zcard(self, name: str | bytes) -> int: ...

    async def zrange(self, name: str | bytes, start: int, end: int) -> list[bytes | str]: ...

    async def zrem(self, name: str | bytes, *members: str | bytes) -> Any: ...


def pack_embedding_vector(vector: list[float]) -> bytes:
    dim = len(vector)
    body = array.array("f", vector).tobytes()
    return _HEADER.pack(_MAGIC, dim) + body


def unpack_embedding_vector(raw: bytes | str | None) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    if len(raw) < _HEADER.size:
        return None
    magic, dim = _HEADER.unpack_from(raw, 0)
    if magic != _MAGIC or dim < 1:
        return None
    expected = _HEADER.size + dim * 4
    if len(raw) != expected:
        return None
    values = array.array("f")
    values.frombytes(raw[_HEADER.size :])
    return values.tolist()


def embedding_field_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_bucket_key(language: str, kind: EmbeddingKind) -> str:
    lang = resolve_search_language(language)
    if kind not in ("query", "tag"):
        raise ValueError(f"unsupported embedding kind: {kind}")
    return f"{EMBEDDING_TEXT_KEY_PREFIX}:{lang}:{kind}"


def _lru_index_key(language: str, kind: EmbeddingKind) -> str:
    return f"{embedding_bucket_key(language, kind)}:_lru"


class EmbeddingTextCache:
    """语言桶 Hash + 每桶 LRU；field 仅为文本 sha256。"""

    def __init__(
        self,
        redis: RedisLike,
        *,
        max_size: int = 4096,
        ttl_sec: int = _WEEK_SECONDS,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if ttl_sec < 1:
            raise ValueError("ttl_sec must be >= 1")
        self._redis = redis
        self._max_size = max_size
        self._ttl_sec = ttl_sec

    async def get(
        self,
        text: str,
        *,
        language: str,
        kind: EmbeddingKind,
    ) -> list[float] | None:
        text = text.strip()
        if not text:
            return None
        bucket = embedding_bucket_key(language, kind)
        field = embedding_field_hash(text)
        try:
            raw = await self._redis.hget(bucket, field)
        except Exception as exc:
            logger.warning("读取 embedding 文本缓存失败：{}", exc)
            return None
        vector = unpack_embedding_vector(raw)
        if vector is None:
            return None
        try:
            await self._bump_lru(language, kind, field)
        except Exception as exc:
            logger.warning("更新 embedding LRU 失败：{}", exc)
        return vector

    async def mget(
        self,
        texts: list[str],
        *,
        language: str,
        kind: EmbeddingKind,
    ) -> list[list[float] | None]:
        if not texts:
            return []
        bucket = embedding_bucket_key(language, kind)
        fields = [embedding_field_hash(text.strip()) for text in texts]
        try:
            raws = await self._redis.hmget(bucket, fields)
        except Exception as exc:
            logger.warning("批量读取 embedding 文本缓存失败：{}", exc)
            return [None] * len(texts)
        out: list[list[float] | None] = []
        for field, raw in zip(fields, raws, strict=True):
            vector = unpack_embedding_vector(raw)
            out.append(vector)
            if vector is None:
                continue
            try:
                await self._bump_lru(language, kind, field)
            except Exception as exc:
                logger.warning("更新 embedding LRU 失败：{}", exc)
        return out

    async def set(
        self,
        text: str,
        vector: list[float],
        *,
        language: str,
        kind: EmbeddingKind,
    ) -> bool:
        text = text.strip()
        if not text or not vector:
            return False
        bucket = embedding_bucket_key(language, kind)
        field = embedding_field_hash(text)
        payload = pack_embedding_vector(vector)
        try:
            await self._redis.hset(bucket, field, payload)
            await self._redis.expire(bucket, self._ttl_sec)
            await self._bump_lru(language, kind, field)
            await self._evict_if_needed(language, kind)
        except Exception as exc:
            logger.warning("写入 embedding 文本缓存失败：{}", exc)
            return False
        return True

    async def delete(
        self,
        text: str,
        *,
        language: str,
        kind: EmbeddingKind,
    ) -> None:
        text = text.strip()
        if not text:
            return
        bucket = embedding_bucket_key(language, kind)
        field = embedding_field_hash(text)
        lru_key = _lru_index_key(language, kind)
        try:
            await self._redis.hdel(bucket, field)
            await self._redis.zrem(lru_key, field)
        except Exception as exc:
            logger.warning("删除 embedding 文本缓存失败：{}", exc)

    async def _bump_lru(self, language: str, kind: EmbeddingKind, field: str) -> None:
        await self._redis.zadd(_lru_index_key(language, kind), {field: time.time()})

    async def _evict_if_needed(self, language: str, kind: EmbeddingKind) -> None:
        lru_key = _lru_index_key(language, kind)
        bucket = embedding_bucket_key(language, kind)
        size = await self._redis.zcard(lru_key)
        overflow = size - self._max_size
        if overflow <= 0:
            return
        victims = await self._redis.zrange(lru_key, 0, overflow - 1)
        if not victims:
            return
        await self._redis.hdel(bucket, *victims)
        await self._redis.zrem(lru_key, *victims)
