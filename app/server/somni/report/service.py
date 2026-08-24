"""量产睡眠报告：按 uid + record_date 查询（实现待补）。"""

from __future__ import annotations


class ReportService:
    async def get_summary(self, uid: str, record_date: str) -> None:
        del uid, record_date

    async def get_events(self, uid: str, record_date: str) -> None:
        del uid, record_date

    async def get_environment(self, uid: str, record_date: str) -> None:
        del uid, record_date

    async def get_structure(self, uid: str, record_date: str) -> None:
        del uid, record_date

    async def get_sleep_quality(self, uid: str, record_date: str) -> None:
        del uid, record_date
