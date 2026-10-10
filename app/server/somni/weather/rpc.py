"""量产天气 gRPC 适配。"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from app.core.exceptions import ServiceNotReadyError
from app.server.errors import abort_from_app_error, abort_invalid, run_rpc_call
from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2, uburnode_somni_pb2_grpc

if TYPE_CHECKING:
    from app.server.somni.weather.service import WeatherService


class WeatherRpc(uburnode_somni_pb2_grpc.WeatherServiceServicer):
    def __init__(self, service: WeatherService | None) -> None:
        self._service = service

    async def GetWeather(self, request, context):
        record_date = request.record_date.strip()
        uid = request.uid.strip()
        timezone = request.timezone.strip()
        if not record_date or not uid or not timezone:
            await abort_invalid(context, "record_date / uid / timezone 均不能为空")
        try:
            date.fromisoformat(record_date)
            ZoneInfo(timezone)
        except Exception:
            await abort_invalid(context, "record_date 或 timezone 非法")
        service = await self._require(context)

        async def _do():
            payload = await service.get_weather(record_date, uid, timezone)
            return to_weather_res(payload)

        return await run_rpc_call(context, _do)

    async def _require(self, context) -> WeatherService:
        if self._service is None:
            await abort_from_app_error(context, ServiceNotReadyError())
        return self._service  # type: ignore[return-value]


def to_weather_message(item: dict[str, Any]) -> uburnode_somni_pb2.Weather:
    temp = item.get("temperature") or {}
    uv = item.get("uv_index") or {}
    return uburnode_somni_pb2.Weather(
        date=str(item.get("date") or ""),
        temperature=uburnode_somni_pb2.Temperature(
            min=int(temp.get("min") or 0),
            max=int(temp.get("max") or 0),
        ),
        condition_code=str(item.get("condition_code") or ""),
        sunrise_at=str(item.get("sunrise_at") or ""),
        sunset_at=str(item.get("sunset_at") or ""),
        uv_index=uburnode_somni_pb2.UvIndex(level=str(uv.get("level") or "")),
        humidity_percent=int(item.get("humidity_percent") or 0),
        aqi=str(item.get("aqi") or ""),
    )


def to_weather_res(payload: dict[str, Any]) -> uburnode_somni_pb2.GetWeatherRes:
    return uburnode_somni_pb2.GetWeatherRes(
        weather=to_weather_message(payload.get("weather") or {})
    )
