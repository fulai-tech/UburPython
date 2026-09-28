"""量产天气服务：本期固定样例。"""

from __future__ import annotations

from typing import Any

from app.server.somni.weather.fixed import fixed_weather


class WeatherService:
    async def get_weather(
        self, record_date: str, uid: str, timezone: str
    ) -> dict[str, Any]:
        _ = uid, timezone
        return {"weather": fixed_weather(record_date)}
