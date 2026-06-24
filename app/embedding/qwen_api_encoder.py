"""DashScope / Qwen OpenAI-compatible embedding API backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiohttp
from loguru import logger

from app.core.exceptions import EncoderNotReadyError
from app.embedding.encoder import EncoderBase

if TYPE_CHECKING:
    from app.core.config import Settings


class QwenApiEncoder(EncoderBase):
    """文本 → 向量（DashScope OpenAI-compatible /embeddings API）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loaded = False

    def load(self) -> None:
        if not self._settings.dashscope_api_key:
            msg = "DASHSCOPE_API_KEY 未配置，无法使用 EMBEDDING_BACKEND=qwen_api"
            raise RuntimeError(msg)
        self._loaded = True
        logger.info(
            "Qwen embedding API 已配置，model={}，dim={}，base_url={}",
            self._settings.embedding_model,
            self._settings.embedding_dim,
            self._settings.dashscope_base_url,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded and bool(self._settings.dashscope_api_key)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.is_loaded:
            raise EncoderNotReadyError()

        vectors: list[list[float]] = []
        batch_size = max(1, self._settings.embedding_api_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def encode_one(self, text: str) -> list[float]:
        results = await self.encode([text])
        return results[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = _join_url(self._settings.dashscope_base_url, "embeddings")
        timeout = aiohttp.ClientTimeout(total=self._settings.embedding_api_timeout_sec)
        payload: dict[str, Any] = {
            "model": self._settings.embedding_model,
            "input": texts,
            "dimensions": self._settings.embedding_dim,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                body = await response.json()
                if response.status >= 400:
                    msg = body.get("message") or body.get("error") or str(body)
                    raise RuntimeError(f"Qwen embedding API 调用失败：HTTP {response.status} {msg}")

        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Qwen embedding API 响应缺少 data：{body}")

        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(vector, list) for vector in vectors):
            raise RuntimeError(f"Qwen embedding API 响应缺少 embedding：{body}")
        return vectors


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
