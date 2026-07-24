"""请求日志中间件：应记录方法/路径，并完整记录请求体（不截断）。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from loguru import logger

from app.middleware.request_log import register_request_log_middleware


def _app_with_echo() -> FastAPI:
    app = FastAPI()
    register_request_log_middleware(app)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"len": len(body)}

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "1"}

    return app


def _app_with_inner_log() -> FastAPI:
    app = FastAPI()
    register_request_log_middleware(app)

    @app.get("/inner")
    async def inner() -> dict[str, str]:
        logger.info("业务层探测日志")
        return {"ok": "1"}

    return app


def test_post_request_logs_body_and_keeps_body_for_handler() -> None:
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)))
    try:
        client = TestClient(_app_with_echo())
        response = client.post("/echo", json={"query_text": "雨声", "top_k": 3})
    finally:
        logger.remove(handler_id)

    assert response.status_code == 200
    assert response.json()["len"] > 0
    joined = "\n".join(messages)
    assert "请求开始" in joined
    assert "请求体=" in joined
    assert "雨声" in joined


def test_get_request_logs_empty_body() -> None:
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)))
    try:
        client = TestClient(_app_with_echo())
        response = client.get("/ping")
    finally:
        logger.remove(handler_id)

    assert response.status_code == 200
    start_lines = [m for m in messages if "请求开始" in m]
    assert start_lines
    assert "请求体=" in start_lines[0]


def test_long_body_is_logged_in_full() -> None:
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)))
    huge = "x" * 4096
    try:
        client = TestClient(_app_with_echo())
        response = client.post("/echo", content=huge.encode("utf-8"))
    finally:
        logger.remove(handler_id)

    assert response.status_code == 200
    assert response.json()["len"] == len(huge)
    start_lines = [m for m in messages if "请求开始" in m]
    assert start_lines
    assert "已截断" not in start_lines[0]
    assert huge in start_lines[0]


def test_response_sets_x_request_id_header() -> None:
    client = TestClient(_app_with_echo())
    response = client.get("/ping", headers={"X-Request-Id": "client-rid-001"})
    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "client-rid-001"


def test_request_id_propagates_to_handler_logs() -> None:
    records: list[dict] = []

    def _capture(message) -> None:
        records.append(message.record)

    handler_id = logger.add(_capture)
    try:
        client = TestClient(_app_with_inner_log())
        response = client.get("/inner", headers={"X-Request-Id": "prop-rid-xyz"})
    finally:
        logger.remove(handler_id)

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "prop-rid-xyz"
    business = [r for r in records if "业务层探测日志" in r["message"]]
    assert business
    assert business[0]["extra"].get("request_id") == "prop-rid-xyz"
    starts = [r for r in records if "请求开始" in r["message"]]
    assert starts
    assert starts[0]["extra"].get("request_id") == "prop-rid-xyz"
