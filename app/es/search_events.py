"""量产音频搜索事件 ES 明细。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from app.core.config import Settings

_MAPPING = {
    "mappings": {
        "properties": {
            "keyword": {"type": "keyword"},
            "raw_query": {"type": "keyword"},
            "created_at": {"type": "date"},
            "hit_count": {"type": "integer"},
            "request_id": {"type": "keyword"},
        }
    }
}


def _build_event_doc(
    *,
    keyword: str,
    raw_query: str,
    hit_count: int,
    request_id: str,
) -> dict[str, Any]:
    return {
        "keyword": keyword,
        "raw_query": raw_query,
        "created_at": datetime.now(UTC).isoformat(),
        "hit_count": int(hit_count),
        "request_id": request_id,
    }


class SearchEventsStore:
    def __init__(self, client: Any, settings: Settings) -> None:
        self._client = client
        self._index = settings.somni_es_search_events_index
        self._ensure_lock = asyncio.Lock()
        self._index_ready = False

    async def ensure_index(self) -> None:
        if self._index_ready:
            return
        async with self._ensure_lock:
            if self._index_ready:
                return
            if await self._client.indices.exists(index=self._index):
                self._index_ready = True
                return
            try:
                await self._client.indices.create(index=self._index, body=_MAPPING)
                logger.info("已创建 ES 索引：{}", self._index)
            except Exception as exc:
                if not _is_already_exists_error(exc):
                    raise
            self._index_ready = True

    async def index_event(
        self,
        *,
        keyword: str,
        raw_query: str,
        hit_count: int,
        request_id: str = "",
    ) -> None:
        await self.ensure_index()
        doc = _build_event_doc(
            keyword=keyword,
            raw_query=raw_query,
            hit_count=hit_count,
            request_id=request_id,
        )
        await self._client.index(index=self._index, document=doc)


def _is_already_exists_error(exc: Exception) -> bool:
    details = (
        str(exc),
        str(getattr(exc, "error", "")),
        str(getattr(exc, "body", "")),
        str(getattr(exc, "info", "")),
    )
    return any(
        marker in detail
        for detail in details
        for marker in ("resource_already_exists_exception", "index_already_exists_exception")
    )
