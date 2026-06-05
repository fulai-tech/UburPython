"""统一 API 信封与异常响应测试（code = HTTP 状态码）。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.codes import HttpStatus
from app.core.exception_handlers import register_exception_handlers
from app.core.exceptions import (
    CommMaterialNotFoundError,
    EncoderNotReadyError,
    ServiceNotReadyError,
)


def _assert_envelope(body: dict, expected_code: int) -> None:
    assert body["code"] == expected_code
    assert "msg" in body
    assert "data" in body
    assert "timestamp" in body
    assert isinstance(body["data"], dict)
    assert isinstance(body["timestamp"], int)


def _client_for(exc: Exception) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise")
    async def _raise() -> None:
        raise exc

    return TestClient(app)


def test_comm_material_not_found_returns_502() -> None:
    response = _client_for(CommMaterialNotFoundError("测试名")).get("/raise")
    assert response.status_code == 502
    _assert_envelope(response.json(), HttpStatus.BAD_GATEWAY)


def test_encoder_not_ready_returns_503() -> None:
    response = _client_for(EncoderNotReadyError()).get("/raise")
    assert response.status_code == 503
    _assert_envelope(response.json(), HttpStatus.SERVICE_UNAVAILABLE)


def test_service_not_ready_returns_503() -> None:
    response = _client_for(ServiceNotReadyError()).get("/raise")
    body = response.json()
    assert response.status_code == 503
    _assert_envelope(body, HttpStatus.SERVICE_UNAVAILABLE)


def test_create_audio_encoder_error_returns_envelope() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.main import create_app

    mock_service = MagicMock()
    mock_service.create_audio = AsyncMock(side_effect=EncoderNotReadyError())
    mock_state = MagicMock()
    mock_state.audio_service = mock_service

    app = create_app()
    with patch("app.main.get_app_state", return_value=mock_state):
        response = TestClient(app).post(
            "/api/audio",
            json={
                "audio_url": "https://cdn.example.com/a.mp3",
                "audio_name": "测试",
            },
        )

    assert response.status_code == 503
    _assert_envelope(response.json(), HttpStatus.SERVICE_UNAVAILABLE)
    mock_service.create_audio.assert_awaited_once()


def test_validation_error_returns_422() -> None:
    from pydantic import BaseModel

    app = FastAPI()
    register_exception_handlers(app)

    class _Body(BaseModel):
        audio_url: str

    @app.post("/validate")
    async def _validate(body: _Body) -> dict[str, str]:
        return {"ignored": "true"}

    response = TestClient(app).post("/validate", json={})
    assert response.status_code == 422
    body = response.json()
    _assert_envelope(body, HttpStatus.UNPROCESSABLE_ENTITY)
    assert "required" in body["msg"].lower()


def test_success_uses_http_200() -> None:
    from app.schemas.response import success

    payload = success(data={"id": "abc"}, msg="创建成功")
    assert payload["code"] == HttpStatus.OK
    assert payload["msg"] == "创建成功"
    assert payload["data"] == {"id": "abc"}
