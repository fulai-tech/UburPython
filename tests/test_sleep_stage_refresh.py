"""睡眠阶段候选缓存延时去抖重建测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.cache.sleep_stage_refresh import DebouncedSleepStageCacheRefresh


@pytest.mark.asyncio
async def test_invalidate_clears_immediately_and_warms_once_after_delay() -> None:
    clear = AsyncMock()
    warm = AsyncMock()
    refresh = DebouncedSleepStageCacheRefresh(clear=clear, warm=warm, delay_sec=0.05)

    await refresh.invalidate()
    clear.assert_awaited_once()
    warm.assert_not_awaited()

    await refresh.invalidate()
    await refresh.invalidate()
    assert clear.await_count == 3

    await asyncio.sleep(0.12)
    warm.assert_awaited_once()


@pytest.mark.asyncio
async def test_later_invalidate_resets_warm_timer() -> None:
    clear = AsyncMock()
    warm = AsyncMock()
    refresh = DebouncedSleepStageCacheRefresh(clear=clear, warm=warm, delay_sec=0.08)

    await refresh.invalidate()
    await asyncio.sleep(0.04)
    await refresh.invalidate()
    await asyncio.sleep(0.05)
    warm.assert_not_awaited()
    await asyncio.sleep(0.06)
    warm.assert_awaited_once()
