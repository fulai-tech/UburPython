"""睡眠阶段候选 Redis 缓存单元测试（文档单份 + 阶段索引）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import pytest

from app.cache.sleep_stage_cache import (
    SLEEP_STAGE_DOC_KEY_PREFIX,
    SLEEP_STAGE_INDEX_KEY_PREFIX,
    SLEEP_STAGES,
    SleepStageCandidateCache,
    build_sleep_stage_doc_key,
    build_sleep_stage_index_key,
    merge_urls_preserve_order,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[bytes, bytes] = {}
        self.ttl: dict[bytes, int] = {}

    def _kb(self, key: str | bytes) -> bytes:
        return key if isinstance(key, bytes) else key.encode()

    async def get(self, key: str | bytes) -> bytes | None:
        return self.store.get(self._kb(key))

    async def set(self, key: str | bytes, value: str | bytes, ex: int | None = None) -> None:
        kb = self._kb(key)
        vb = value if isinstance(value, bytes) else value.encode()
        self.store[kb] = vb
        if ex is not None:
            self.ttl[kb] = ex

    async def mget(self, keys: list[str | bytes]) -> list[bytes | None]:
        return [self.store.get(self._kb(key)) for key in keys]

    async def delete(self, *keys: str | bytes) -> int:
        deleted = 0
        for key in keys:
            kb = self._kb(key)
            if kb in self.store:
                del self.store[kb]
                deleted += 1
            self.ttl.pop(kb, None)
        return deleted


def _doc(name: str, url: str, stages: list[str]) -> dict[str, Any]:
    return {
        "audio_name": name,
        "audio_url": url,
        "sleep_stage_tags": [{"tag_id": f"s_{s}", "code": s, "name": s} for s in stages],
    }


def test_key_builders() -> None:
    assert SLEEP_STAGE_INDEX_KEY_PREFIX == "sleep_stage_v2_index:"
    assert SLEEP_STAGE_DOC_KEY_PREFIX == "sleep_stage_v2_doc:"
    assert build_sleep_stage_index_key("放松") == f"{SLEEP_STAGE_INDEX_KEY_PREFIX}放松"
    assert build_sleep_stage_doc_key("https://cdn/a.mp3").startswith(SLEEP_STAGE_DOC_KEY_PREFIX)
    assert SLEEP_STAGES == ("放松", "入睡", "守护", "唤醒")


def test_merge_urls_preserve_order() -> None:
    assert merge_urls_preserve_order([["https://a", "https://b"], ["https://a", "https://c"]]) == [
        "https://a",
        "https://b",
        "https://c",
    ]


@pytest.mark.asyncio
async def test_set_stage_stores_shared_doc_once() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    shared = _doc("共享", "https://cdn/share.mp3", ["放松", "入睡"])

    await cache.set_stage("放松", [shared])
    await cache.set_stage("入睡", [shared])

    doc_keys = [k for k in redis.store if k.startswith(SLEEP_STAGE_DOC_KEY_PREFIX.encode())]
    assert len(doc_keys) == 1
    assert redis.ttl[doc_keys[0]] == 60
    index_relax = json.loads(redis.store[build_sleep_stage_index_key("放松").encode()])
    index_sleep = json.loads(redis.store[build_sleep_stage_index_key("入睡").encode()])
    assert index_relax == ["https://cdn/share.mp3"]
    assert index_sleep == ["https://cdn/share.mp3"]


@pytest.mark.asyncio
async def test_set_and_get_single_stage() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    docs = [_doc("雨声", "https://cdn/rain.mp3", ["入睡"])]

    await cache.set_stage("入睡", docs)
    got = await cache.get(["入睡"])

    assert got == docs


@pytest.mark.asyncio
async def test_get_multi_stage_merges_and_dedupes_by_audio_url() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    shared = _doc("共享", "https://cdn/share.mp3", ["放松", "入睡"])
    only_relax = _doc("仅放松", "https://cdn/relax.mp3", ["放松"])
    only_sleep = _doc("仅入睡", "https://cdn/sleep.mp3", ["入睡"])

    await cache.set_stage("放松", [shared, only_relax])
    await cache.set_stage("入睡", [shared, only_sleep])

    got = await cache.get(["放松", "入睡"])

    assert [d["audio_url"] for d in got] == [
        "https://cdn/share.mp3",
        "https://cdn/relax.mp3",
        "https://cdn/sleep.mp3",
    ]
    doc_keys = [k for k in redis.store if k.startswith(SLEEP_STAGE_DOC_KEY_PREFIX.encode())]
    assert len(doc_keys) == 3


@pytest.mark.asyncio
async def test_get_returns_none_when_any_stage_index_missing() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    await cache.set_stage("放松", [_doc("A", "https://cdn/a.mp3", ["放松"])])

    assert await cache.get(["放松", "入睡"]) is None


@pytest.mark.asyncio
async def test_get_returns_none_when_doc_missing() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    await cache.set_stage("放松", [_doc("A", "https://cdn/a.mp3", ["放松"])])
    await redis.delete(build_sleep_stage_doc_key("https://cdn/a.mp3"))

    assert await cache.get(["放松"]) is None


@pytest.mark.asyncio
async def test_warm_loads_each_known_stage_from_loader() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    calls: list[str] = []

    async def loader(stage: str) -> list[dict[str, Any]]:
        calls.append(stage)
        return [_doc(stage, f"https://cdn/{stage}.mp3", [stage])]

    await cache.warm(loader)

    assert calls == list(SLEEP_STAGES)
    for stage in SLEEP_STAGES:
        raw = await redis.get(build_sleep_stage_index_key(stage))
        assert raw is not None
        assert json.loads(raw) == [f"https://cdn/{stage}.mp3"]


@pytest.mark.asyncio
async def test_get_or_load_only_populates_missing_stage() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    relax = _doc("放松", "https://cdn/relax.mp3", ["放松"])
    wake = _doc("唤醒", "https://cdn/wake.mp3", ["唤醒"])
    await cache.set_stage("放松", [relax])
    calls: list[str] = []

    async def loader(stage: str) -> list[dict[str, Any]]:
        calls.append(stage)
        return [wake]

    got = await cache.get_or_load(["放松", "唤醒"], loader)

    assert calls == ["唤醒"]
    assert [doc["audio_url"] for doc in got] == [
        "https://cdn/relax.mp3",
        "https://cdn/wake.mp3",
    ]
    assert await cache.get(["放松"]) == [relax]


@pytest.mark.asyncio
async def test_get_or_load_coalesces_concurrent_stage_misses() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    wake = _doc("唤醒", "https://cdn/wake.mp3", ["唤醒"])
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def loader(stage: str) -> list[dict[str, Any]]:
        nonlocal calls
        assert stage == "唤醒"
        calls += 1
        started.set()
        await release.wait()
        return [wake]

    first = asyncio.create_task(cache.get_or_load(["唤醒"], loader))
    await started.wait()
    second = asyncio.create_task(cache.get_or_load(["唤醒"], loader))
    release.set()

    assert await asyncio.gather(first, second) == [[wake], [wake]]
    assert calls == 1


@pytest.mark.asyncio
async def test_get_or_load_caches_empty_stage() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    calls = 0

    async def loader(stage: str) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return []

    assert await cache.get_or_load(["唤醒"], loader) == []
    assert await cache.get_or_load(["唤醒"], loader) == []
    assert calls == 1


@pytest.mark.asyncio
async def test_get_or_load_does_not_cache_loader_error() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)

    async def failing_loader(stage: str) -> list[dict[str, Any]]:
        raise RuntimeError("ES unavailable")

    with pytest.raises(RuntimeError, match="ES unavailable"):
        await cache.get_or_load(["唤醒"], failing_loader)

    assert await cache.get(["唤醒"]) is None


@pytest.mark.asyncio
async def test_clear_waits_for_inflight_stage_load() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    started = asyncio.Event()
    release = asyncio.Event()

    async def loader(stage: str) -> list[dict[str, Any]]:
        started.set()
        await release.wait()
        return [_doc(stage, "https://cdn/wake.mp3", [stage])]

    load_task = asyncio.create_task(cache.get_or_load(["唤醒"], loader))
    await started.wait()
    clear_task = asyncio.create_task(cache.clear_all())
    await asyncio.sleep(0)
    assert not clear_task.done()

    release.set()
    await load_task
    await clear_task

    assert redis.store == {}


@pytest.mark.asyncio
async def test_clear_all_removes_indexes_and_docs() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    for stage in SLEEP_STAGES:
        await cache.set_stage(stage, [_doc(stage, f"https://cdn/{stage}.mp3", [stage])])

    await cache.clear_all()

    assert redis.store == {}


@pytest.mark.asyncio
async def test_clear_all_removes_legacy_indexes_docs_and_candidate_keys() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    url = "https://cdn/legacy.mp3"
    legacy_doc_key = f"sleep_stage_doc:{hashlib.sha256(url.encode('utf-8')).hexdigest()}"

    await redis.set("sleep_stage_index:放松", json.dumps([url]), ex=60)
    await redis.set(legacy_doc_key, json.dumps(_doc("旧文档", url, ["放松"])), ex=60)
    await redis.set("sleep_stage_candidates:放松", json.dumps([{"audio_url": url}]), ex=60)

    await cache.clear_all()

    assert await redis.get("sleep_stage_index:放松") is None
    assert await redis.get(legacy_doc_key) is None
    assert await redis.get("sleep_stage_candidates:放松") is None


@pytest.mark.asyncio
async def test_clear_all_removes_stale_awake_stage_index() -> None:
    redis = _FakeRedis()
    cache = SleepStageCandidateCache(redis, ttl_sec=60)
    await cache.set_stage("清醒", [_doc("旧清醒", "https://cdn/awake.mp3", ["清醒"])])

    await cache.clear_all()

    assert await redis.get(build_sleep_stage_index_key("清醒")) is None
