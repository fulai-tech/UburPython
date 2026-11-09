"""天气 protobuf 映射（供 AppleWatch GetDailyAggregate 复用）。"""

from __future__ import annotations

from typing import Any

from app.uburnode_grpc.grpc_gen import uburnode_somni_pb2


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
