"""统一 HTTP 响应信封：{ code, msg, data, timestamp }，code 与 HTTP 状态码一致。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.core.codes import HttpStatus


class ApiResponse(BaseModel):
    code: int = Field(description="与 HTTP 状态码一致，200 为成功")
    msg: str = Field(description="提示信息")
    data: dict[str, Any] = Field(default_factory=dict, description="业务数据")
    timestamp: int = Field(description="Unix 毫秒时间戳")


def current_timestamp_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def build_response(
    *,
    code: int,
    msg: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ApiResponse(
        code=code,
        msg=msg,
        data=data or {},
        timestamp=current_timestamp_ms(),
    ).model_dump()


def success(
    data: dict[str, Any] | None = None,
    msg: str = "success",
    code: int = HttpStatus.OK,
) -> dict[str, Any]:
    return build_response(code=code, msg=msg, data=data)


def fail(
    code: int,
    msg: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_response(code=code, msg=msg, data=data)
