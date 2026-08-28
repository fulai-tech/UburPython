"""量产音频搜索热点：Redis 周榜计数 + ES 明细。

kind=query：用户搜索词周榜
kind=tag：内容形态标签 code 周榜
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from loguru import logger

from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError
from app.es.search_events import SearchEventsStore

HotKind = Literal["query", "tag"]
_HOT_KINDS = frozenset({"query", "tag"})


def normalize_keyword(text: str) -> str:
    """去首尾空白、压缩中间空白、英文大小写归一（casefold）。"""
    return " ".join(text.strip().casefold().split())


def normalize_tag_code(code: str) -> str:
    """标签 code 归一：trim + casefold。"""
    return code.strip().casefold()


def parse_hot_kind(raw: str | None) -> HotKind:
    value = (raw or "").strip().lower() or "query"
    if value not in _HOT_KINDS:
        raise ValueError(f"kind 仅支持 query / tag，当前为 {raw!r}")
    return value  # type: ignore[return-value]


def hot_week_id(now: datetime | None = None) -> str:
    """ISO 周标识，如 2026-W35。"""
    dt = now or datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def hot_redis_key(
    settings: Settings,
    language: str,
    *,
    kind: HotKind = "query",
    week_id: str | None = None,
) -> str:
    """按 kind + 语言 + ISO 周分桶。"""
    return f"{settings.somni_hot_redis_key}:{kind}:{language}:{week_id or hot_week_id()}"


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

    async def record_search(
        self,
        raw_query: str,
        *,
        language: str,
        hit_count: int,
        tag_codes: set[str] | None = None,
    ) -> None:
        if not self._settings.somni_hot_enabled:
            return
        keyword = normalize_keyword(raw_query)
        codes = {normalize_tag_code(c) for c in (tag_codes or set()) if c and c.strip()}
        if keyword:
            # 排行只统计有结果的搜索；ES 仍留全量明细
            if hit_count > 0:
                await self._safe_redis_incr(keyword, language, kind="query")
            await self._safe_es_index(
                keyword,
                raw_query,
                language,
                hit_count,
                kind="query",
            )
        if hit_count > 0 and codes:
            for code in codes:
                await self._safe_redis_incr(code, language, kind="tag")
                await self._safe_es_index(
                    code,
                    raw_query or code,
                    language,
                    hit_count,
                    kind="tag",
                )

    async def list_hot(
        self,
        *,
        language: str,
        kind: HotKind = "query",
    ) -> list[dict[str, Any]]:
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
                hot_redis_key(self._settings, language, kind=kind),
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

    async def _safe_redis_incr(
        self,
        keyword: str,
        language: str,
        *,
        kind: HotKind,
    ) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.zincrby(
                hot_redis_key(self._settings, language, kind=kind),
                1,
                keyword,
            )
        except Exception as exc:
            logger.warning("量产热点 Redis 写入失败：{}", exc)

    async def _safe_es_index(
        self,
        keyword: str,
        raw_query: str,
        language: str,
        hit_count: int,
        *,
        kind: HotKind,
    ) -> None:
        if self._events is None:
            return
        try:
            await self._events.index_event(
                keyword=keyword,
                raw_query=raw_query,
                language=language,
                hit_count=hit_count,
                kind=kind,
            )
        except Exception as exc:
            logger.warning("量产热点 ES 写入失败：{}", exc)
