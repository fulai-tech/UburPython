"""PyTorch / sentence-transformers 向量编码（开发对比与回退用）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
from loguru import logger

from app.core.exceptions import EncoderNotReadyError
from app.embedding.encoder import EncoderBase

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from app.core.config import Settings


class TorchEncoder(EncoderBase):
    """文本 → 512 维向量（BAAI/bge-small-zh-v1.5，normalize_embeddings=True）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("正在加载 PyTorch 向量模型：{}", self._settings.embedding_model)
        self._model = SentenceTransformer(self._settings.embedding_model)
        logger.info("PyTorch 向量模型加载完成")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def encode(
        self, texts: list[str], *, language: str = "zh"
    ) -> list[list[float]]:
        del language
        if not texts:
            return []
        if self._model is None:
            raise EncoderNotReadyError()
        return await anyio.to_thread.run_sync(self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    async def encode_one(self, text: str, *, language: str = "zh") -> list[float]:
        results = await self.encode([text], language=language)
        return results[0]
