"""按睡眠阶段拆分的候选文档 Redis 缓存。

文档全局只存一份（sleep_stage_v2_doc:*），各阶段仅存 audio_url 索引；
多阶段查询合并索引后 MGET 文档，按 audio_url 去重。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from loguru import logger

from app.core.config import Settings

SLEEP_STAGES = ("放松", "入睡", "守护", "清醒")
SLEEP_STAGE_INDEX_KEY_PREFIX = "sleep_stage_v2_index:"
SLEEP_STAGE_DOC_KEY_PREFIX = "sleep_stage_v2_doc:"
# 兼容旧测试/调用名
SLEEP_STAGE_CANDIDATE_KEY_PREFIX = SLEEP_STAGE_INDEX_KEY_PREFIX
_LEGACY_SLEEP_STAGE_INDEX_KEY_PREFIX = "sleep_stage_index:"
_LEGACY_SLEEP_STAGE_DOC_KEY_PREFIX = "sleep_stage_doc:"
_LEGACY_SLEEP_STAGE_CANDIDATE_KEY_PREFIX = "sleep_stage_candidates:"
_WEEK_SECONDS = 7 * 24 * 60 * 60

StageLoader = Callable[[str], Awaitable[list[dict[str, Any]]]]


class RedisLike(Protocol):
    async def get(self, key: str | bytes) -> bytes | None: ...

    async def set(self, key: str | bytes, value: str | bytes, ex: int | None = None) -> Any: ...

    async def mget(self, keys: list[Any]) -> list[bytes | None]: ...

    async def delete(self, *keys: str | bytes) -> Any: ...


def build_sleep_stage_index_key(stage: str) -> str:
    return f"{SLEEP_STAGE_INDEX_KEY_PREFIX}{stage}"


def build_sleep_stage_cache_key(stage: str) -> str:
    """兼容旧名：阶段索引 key。"""
    return build_sleep_stage_index_key(stage)


def build_sleep_stage_doc_key(audio_url: str) -> str:
    digest = hashlib.sha256(audio_url.encode("utf-8")).hexdigest()
    return f"{SLEEP_STAGE_DOC_KEY_PREFIX}{digest}"


def merge_urls_preserve_order(groups: list[list[str]]) -> list[str]:
    """多阶段 url 列表合并去重，保留首次出现顺序。"""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for url in group:
            item = url.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def merge_candidates_by_audio_url(
    groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """兼容旧辅助：按 audio_url 去重合并文档列表。"""
    url_groups: list[list[str]] = []
    by_url: dict[str, dict[str, Any]] = {}
    for group in groups:
        urls: list[str] = []
        for doc in group:
            url = _doc_audio_url(doc)
            if not url:
                continue
            urls.append(url)
            by_url.setdefault(url, doc)
        url_groups.append(urls)
    return [by_url[url] for url in merge_urls_preserve_order(url_groups) if url in by_url]


class SleepStageCandidateCache:
    """文档单份存储 + 四阶段 url 索引。"""

    def __init__(self, redis: RedisLike, *, ttl_sec: int = _WEEK_SECONDS) -> None:
        if ttl_sec < 1:
            raise ValueError("ttl_sec must be >= 1")
        self._redis = redis
        self._ttl_sec = ttl_sec

    async def get(self, stages: list[str]) -> list[dict[str, Any]] | None:
        """全部阶段索引命中且文档齐全才返回；否则 None。"""
        normalized = _normalize_stages(stages)
        if not normalized:
            return []

        url_groups: list[list[str]] = []
        for stage in normalized:
            raw = await self._redis.get(build_sleep_stage_index_key(stage))
            if raw is None:
                logger.info("睡眠阶段候选缓存未命中，stage={}", stage)
                return None
            urls = json.loads(raw)
            if not isinstance(urls, list):
                return None
            url_groups.append([str(url) for url in urls])

        merged_urls = merge_urls_preserve_order(url_groups)
        if not merged_urls:
            logger.info(
                "睡眠阶段候选缓存命中，stages={}，合并后候选数=0",
                normalized,
            )
            return []

        doc_keys = [build_sleep_stage_doc_key(url) for url in merged_urls]
        values = await self._redis.mget(doc_keys)
        docs: list[dict[str, Any]] = []
        for url, raw in zip(merged_urls, values, strict=True):
            if raw is None:
                logger.info("睡眠阶段候选文档缺失，audio_url={}", url)
                return None
            docs.append(json.loads(raw))

        logger.info(
            "睡眠阶段候选缓存命中，stages={}，合并后候选数={}",
            normalized,
            len(docs),
        )
        return docs

    async def set_stage(self, stage: str, docs: list[dict[str, Any]]) -> None:
        """写入该阶段 url 索引，并 upsert 对应文档（同 url 只存一份）。"""
        urls: list[str] = []
        for doc in docs:
            url = _doc_audio_url(doc)
            if not url:
                continue
            payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
            await self._redis.set(build_sleep_stage_doc_key(url), payload, ex=self._ttl_sec)
            urls.append(url)

        index_payload = json.dumps(urls, ensure_ascii=False, separators=(",", ":"))
        await self._redis.set(
            build_sleep_stage_index_key(stage),
            index_payload,
            ex=self._ttl_sec,
        )
        logger.info(
            "睡眠阶段候选缓存已写入，stage={}，索引数={}，文档 upsert={}",
            stage,
            len(urls),
            len(urls),
        )

    async def warm(self, loader: StageLoader) -> None:
        """清空后按四个阶段从数据源重建索引与文档。"""
        await self.clear_all()
        for stage in SLEEP_STAGES:
            docs = await loader(stage)
            await self.set_stage(stage, docs)
        logger.info("睡眠阶段候选缓存预热完成，stages={}", list(SLEEP_STAGES))

    async def clear_all(self) -> None:
        """删除四个阶段索引及其引用的全部文档。"""
        urls: set[str] = set()
        index_keys = [build_sleep_stage_index_key(stage) for stage in SLEEP_STAGES]
        for key in index_keys:
            raw = await self._redis.get(key)
            if raw is None:
                continue
            items = json.loads(raw)
            if isinstance(items, list):
                urls.update(str(item) for item in items if item)

        legacy_urls: set[str] = set()
        legacy_index_keys = [
            f"{_LEGACY_SLEEP_STAGE_INDEX_KEY_PREFIX}{stage}" for stage in SLEEP_STAGES
        ]
        for key in legacy_index_keys:
            raw = await self._redis.get(key)
            if raw is None:
                continue
            items = json.loads(raw)
            if isinstance(items, list):
                legacy_urls.update(str(item) for item in items if item)

        doc_keys = [build_sleep_stage_doc_key(url) for url in urls]
        legacy_doc_keys = [
            _build_legacy_sleep_stage_doc_key(url) for url in legacy_urls
        ]
        legacy_candidate_keys = [
            f"{_LEGACY_SLEEP_STAGE_CANDIDATE_KEY_PREFIX}{stage}" for stage in SLEEP_STAGES
        ]
        to_delete = [
            *index_keys,
            *doc_keys,
            *legacy_index_keys,
            *legacy_doc_keys,
            *legacy_candidate_keys,
        ]
        if to_delete:
            await self._redis.delete(*to_delete)
        logger.info(
            "已清除睡眠阶段候选缓存，索引={}，文档={}",
            len(index_keys) + len(legacy_index_keys),
            len(doc_keys) + len(legacy_doc_keys),
        )


def _doc_audio_url(doc: dict[str, Any]) -> str:
    return str(doc.get("audio_url", "")).strip()


def _build_legacy_sleep_stage_doc_key(audio_url: str) -> str:
    digest = hashlib.sha256(audio_url.encode("utf-8")).hexdigest()
    return f"{_LEGACY_SLEEP_STAGE_DOC_KEY_PREFIX}{digest}"


def _normalize_stages(stages: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for stage in stages:
        name = stage.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


async def create_sleep_stage_candidate_cache(
    settings: Settings,
    redis: Any | None = None,
) -> SleepStageCandidateCache | None:
    """复用已有 Redis 客户端；否则按 REDIS_URL 新建。空 URL 返回 None。"""
    if redis is not None:
        return SleepStageCandidateCache(redis, ttl_sec=settings.search_cache_ttl_sec)
    if not settings.redis_url.strip():
        logger.warning("未配置 REDIS_URL，睡眠阶段候选缓存已关闭")
        return None
    from redis.asyncio import Redis

    client = Redis.from_url(
        settings.redis_url,
        decode_responses=False,
        max_connections=max(1, settings.redis_max_connections),
    )
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        if settings.app_debug:
            logger.warning("Redis 不可用（调试模式，跳过睡眠阶段缓存）：{}", exc)
            return None
        raise
    return SleepStageCandidateCache(client, ttl_sec=settings.search_cache_ttl_sec)
