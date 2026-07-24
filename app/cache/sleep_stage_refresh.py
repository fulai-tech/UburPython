"""睡眠阶段候选缓存：CUD 后延时去抖重建。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

ClearFn = Callable[[], Awaitable[None]]
WarmFn = Callable[[], Awaitable[None]]


class DebouncedSleepStageCacheRefresh:
    """CUD 触发：立即清空，延迟后重建；窗口内多次调用只重建一次。"""

    def __init__(
        self,
        *,
        clear: ClearFn,
        warm: WarmFn,
        delay_sec: float = 5.0,
    ) -> None:
        if delay_sec < 0:
            raise ValueError("delay_sec must be >= 0")
        self._clear = clear
        self._warm = warm
        self._delay_sec = delay_sec
        self._task: asyncio.Task[None] | None = None
        self._generation = 0

    async def invalidate(self) -> None:
        """立即清缓存，并（重新）预约延时预热。"""
        await self._clear()
        self._generation += 1
        scheduled_gen = self._generation
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = asyncio.create_task(self._delayed_warm(scheduled_gen))
        logger.info(
            "已预约睡眠阶段候选缓存重建，delay_sec={}，generation={}",
            self._delay_sec,
            scheduled_gen,
        )

    async def _delayed_warm(self, generation: int) -> None:
        try:
            if self._delay_sec > 0:
                await asyncio.sleep(self._delay_sec)
            if generation != self._generation:
                return
            await self._warm()
            logger.info("延时重建睡眠阶段候选缓存完成，generation={}", generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("延时重建睡眠阶段候选缓存失败：{}", exc)

    async def flush(self) -> None:
        """测试/关机：取消预约并立即重建（若仍有待执行代际）。"""
        pending_gen = self._generation
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if pending_gen > 0:
            await self._warm()
