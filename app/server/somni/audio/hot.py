"""量产音频搜索热点：Redis 计数 + ES 明细。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError
from app.es.search_events import SearchEventsStore


def normalize_keyword(text: str) -> str:
    return text.strip()


def _as_str(member: bytes | str) -> str:
    if isinstance(member, bytes):
        return member.decode()
    return member


class HotTracker:
    def __init__(
        self,
        redis: Any,
        events_store: SearchEventsStore | None,
        settings: Settings,
    ) -> None:
        self._redis = redis
        self._events = events_store
        self._settings = settings

    async def record_search(self, raw_query: str, *, hit_count: int) -> None:
        if not self._settings.somni_hot_enabled:
            return
        keyword = normalize_keyword(raw_query)
        if not keyword:
            return
        await self._safe_redis_incr(keyword)
        await self._safe_es_index(keyword, raw_query, hit_count)

    async def list_hot(self) -> list[dict[str, Any]]:
        if not self._settings.somni_hot_enabled:
            return []
        if self._settings.somni_hot_top_n <= 0:
            return []
        if self._redis is None:
            raise AppError(
                message="量产 Redis 未配置，无法获取热点",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            )
        try:
            rows = await self._redis.zrevrange(
                self._settings.somni_hot_redis_key,
                0,
                self._settings.somni_hot_top_n - 1,
                withscores=True,
            )
        except Exception as exc:
            logger.warning("量产热点 Redis 读取失败：{}", exc)
            raise AppError(
                message="量产 Redis 不可用，无法获取热点",
                status_code=HttpStatus.SERVICE_UNAVAILABLE,
            ) from exc
        return [{"keyword": _as_str(member), "score": int(score)} for member, score in rows]

    async def _safe_redis_incr(self, keyword: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.zincrby(self._settings.somni_hot_redis_key, 1, keyword)
        except Exception as exc:
            logger.warning("量产热点 Redis 写入失败：{}", exc)

    async def _safe_es_index(
        self,
        keyword: str,
        raw_query: str,
        hit_count: int,
    ) -> None:
        if self._events is None:
            return
        try:
            await self._events.index_event(
                keyword=keyword,
                raw_query=raw_query,
                hit_count=hit_count,
            )
        except Exception as exc:
            logger.warning("量产热点 ES 写入失败：{}", exc)
