"""将统一信封写入 JSONResponse（HTTP 状态码与 body.code 一致）。"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from app.schemas.response import fail, success


def json_success(
    data: dict[str, Any] | None = None,
    msg: str = "success",
    *,
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=success(data=data, msg=msg, code=status_code))


def json_fail(
    code: int,
    msg: str,
    *,
    status_code: int | None = None,
    data: dict[str, Any] | None = None,
) -> JSONResponse:
    http_status = status_code if status_code is not None else code
    return JSONResponse(status_code=http_status, content=fail(code=http_status, msg=msg, data=data))
