"""Qwen / DashScope API embedding backend tests."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.embedding.qwen_api_encoder import QwenApiEncoder


def test_qwen_api_encoder_requires_api_key() -> None:
    encoder = QwenApiEncoder(Settings(embedding_backend="qwen_api", dashscope_api_key=""))

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        encoder.load()


@pytest.mark.asyncio
async def test_qwen_api_encoder_calls_openai_compatible_embeddings(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status = 200

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }

    class FakeSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["session_kwargs"] = kwargs

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("app.embedding.qwen_api_encoder.aiohttp.ClientSession", FakeSession)
    settings = Settings(
        embedding_backend="qwen_api",
        embedding_model="text-embedding-v4",
        embedding_dim=1024,
        dashscope_api_key="sk-test",
        dashscope_base_url="https://dashscope.example/v1/",
        embedding_api_batch_size=10,
    )
    encoder = QwenApiEncoder(settings)
    encoder.load()

    vectors = await encoder.encode(["雨声", "white noise"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "https://dashscope.example/v1/embeddings"
    assert captured["json"] == {
        "model": "text-embedding-v4",
        "input": ["雨声", "white noise"],
        "dimensions": 1024,
    }
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
