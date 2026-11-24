"""embedding 文本向量 Redis 缓存（语言桶 Hash + field sha256）。"""

from __future__ import annotations

import hashlib

import pytest

from app.cache.embedding_text_cache import (
    EmbeddingTextCache,
    embedding_bucket_key,
    embedding_field_hash,
    pack_embedding_vector,
    unpack_embedding_vector,
)


def test_pack_unpack_roundtrip() -> None:
    vector = [0.1, -0.25, 0.5, 1.0]
    raw = pack_embedding_vector(vector)
    assert unpack_embedding_vector(raw) == pytest.approx(vector)


def test_unpack_rejects_corrupt_payload() -> None:
    assert unpack_embedding_vector(b"xx") is None
    assert unpack_embedding_vector(pack_embedding_vector([1.0, 2.0])[:-1]) is None


def test_field_hash_is_sha256_not_chinese() -> None:
    field = embedding_field_hash("雨声")
    assert field == hashlib.sha256("雨声".encode("utf-8")).hexdigest()
    assert field.isascii()


def test_bucket_keys_split_language_and_kind() -> None:
    assert embedding_bucket_key("zh", "query") == "embedding_text:zh:query"
    assert embedding_bucket_key("en", "tag") == "embedding_text:en:tag"
    assert embedding_bucket_key("zh", "query") != embedding_bucket_key("zh", "tag")


class _FakeRedis:
    def __init__(self) -> None:
        self.hash: dict[str, dict[str, bytes]] = {}
        self.zset: dict[str, dict[str, float]] = {}
        self.ttl: dict[str, int] = {}

    async def hget(self, name: str | bytes, key: str | bytes) -> bytes | None:
        return self.hash.get(str(name), {}).get(str(key))

    async def hmget(self, name: str | bytes, keys: list) -> list[bytes | None]:
        bucket = self.hash.get(str(name), {})
        return [bucket.get(str(k)) for k in keys]

    async def hset(self, name: str | bytes, key: str | bytes, value: str | bytes) -> None:
        self.hash.setdefault(str(name), {})[str(key)] = (
            bytes(value) if not isinstance(value, bytes) else value
        )

    async def hdel(self, name: str | bytes, *keys: str | bytes) -> int:
        bucket = self.hash.get(str(name), {})
        n = 0
        for key in keys:
            if bucket.pop(str(key), None) is not None:
                n += 1
        return n

    async def delete(self, *keys: str | bytes) -> int:
        n = 0
        for key in keys:
            if self.hash.pop(str(key), None) is not None:
                n += 1
            if self.zset.pop(str(key), None) is not None:
                n += 1
        return n

    async def expire(self, name: str | bytes, time: int) -> None:
        self.ttl[str(name)] = time

    async def zadd(self, name: str | bytes, mapping: dict) -> None:
        bucket = self.zset.setdefault(str(name), {})
        for member, score in mapping.items():
            bucket[str(member)] = float(score)

    async def zcard(self, name: str | bytes) -> int:
        return len(self.zset.get(str(name), {}))

    async def zrange(self, name: str | bytes, start: int, end: int) -> list[str]:
        items = sorted(self.zset.get(str(name), {}).items(), key=lambda kv: kv[1])
        if end == -1:
            end = len(items) - 1
        return [member for member, _ in items[start : end + 1]]

    async def zrem(self, name: str | bytes, *members: str | bytes) -> None:
        bucket = self.zset.get(str(name), {})
        for member in members:
            bucket.pop(str(member), None)


@pytest.mark.asyncio
async def test_redis_hash_get_set_and_lru_evict() -> None:
    redis = _FakeRedis()
    cache = EmbeddingTextCache(redis, max_size=2, ttl_sec=60)

    assert await cache.set("a", [1.0, 0.0], language="zh", kind="query")
    assert await cache.set("b", [0.0, 1.0], language="zh", kind="query")
    assert await cache.get("a", language="zh", kind="query") == pytest.approx([1.0, 0.0])
    assert await cache.set("c", [0.5, 0.5], language="zh", kind="query")
    assert await cache.get("a", language="zh", kind="query") == pytest.approx([1.0, 0.0])
    assert await cache.get("b", language="zh", kind="query") is None
    assert await cache.get("c", language="zh", kind="query") == pytest.approx([0.5, 0.5])
    # field 为 hash，桶内无中文 field
    fields = set(redis.hash["embedding_text:zh:query"])
    assert all(f.isascii() and len(f) == 64 for f in fields)


@pytest.mark.asyncio
async def test_query_and_tag_use_different_buckets() -> None:
    redis = _FakeRedis()
    cache = EmbeddingTextCache(redis, max_size=16, ttl_sec=60)
    await cache.set("rain", [1.0, 0.0], language="zh", kind="query")
    await cache.set("rain", [0.0, 1.0], language="zh", kind="tag")
    assert await cache.get("rain", language="zh", kind="query") == pytest.approx([1.0, 0.0])
    assert await cache.get("rain", language="zh", kind="tag") == pytest.approx([0.0, 1.0])


@pytest.mark.asyncio
async def test_encode_texts_uses_redis_and_skips_onnx() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from app.core.config import Settings
    from app.es.search import EsSearch
    from app.services.retrieval import RetrievalService

    redis = _FakeRedis()
    cache = EmbeddingTextCache(redis, max_size=16, ttl_sec=60)
    es = MagicMock(spec=EsSearch)
    es.list_content_tag_vectors = AsyncMock(return_value=[])
    encoder = MagicMock()
    encoder.encode = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0, 0.0])
    service = RetrievalService(
        es,
        encoder,
        Settings(embedding_dim=3, embedding_dim_en=3, embedding_text_cache_size=16),
        text_vector_cache=cache,
        use_process_text_vector_cache=False,
    )

    first = await service._encode_texts(["rain"], natural=True, language="zh")
    second = await service._encode_texts(["rain"], natural=True, language="zh")
    assert first == second
    encoder.encode_one.assert_awaited_once()
    encoder.encode.assert_not_awaited()


@pytest.mark.asyncio
async def test_somni_without_redis_does_not_use_process_lru() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from app.core.config import Settings
    from app.es.search import EsSearch
    from app.services.retrieval import RetrievalService

    es = MagicMock(spec=EsSearch)
    es.list_content_tag_vectors = AsyncMock(return_value=[])
    encoder = MagicMock()
    encoder.encode = AsyncMock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts])
    encoder.encode_one = AsyncMock(return_value=[1.0, 0.0])
    service = RetrievalService(
        es,
        encoder,
        Settings(embedding_dim=2, embedding_dim_en=2),
        use_process_text_vector_cache=False,
    )
    await service._encode_texts(["a"], natural=True)
    await service._encode_texts(["a"], natural=True)
    assert encoder.encode_one.await_count == 2
    assert service._text_vector_cache == {}
