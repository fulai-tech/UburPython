"""ONNX Runtime 向量编码（生产默认，无 PyTorch 依赖）。"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import anyio
import numpy as np
import onnxruntime as ort
from loguru import logger
from transformers import AutoTokenizer

from app.core.exceptions import EncoderNotReadyError
from app.embedding.encoder import EncoderBase
from app.embedding.pooling import cls_pool, l2_normalize

if TYPE_CHECKING:
    from app.core.config import Settings


class OnnxEncoder(EncoderBase):
    """文本 → 512 维向量（ONNX 输出 last_hidden_state，Python 侧 CLS pool + L2）。

    多 InferenceSession 池化：同一时刻最多 pool_size 路并发推理，避免单 session 全局锁串行。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: list[ort.InferenceSession] = []
        self._session_pool: asyncio.Queue[ort.InferenceSession] | None = None
        self._pool_init_lock = asyncio.Lock()
        self._tokenizer: AutoTokenizer | None = None
        self._tokenize_lock = threading.Lock()

    def load(self) -> None:
        onnx_path = self._settings.embedding_onnx_path
        tokenizer_dir = self._settings.embedding_tokenizer_dir
        if not onnx_path.is_file():
            msg = f"ONNX 模型不存在：{onnx_path}，请先运行 scripts/export_onnx_model.py"
            raise FileNotFoundError(msg)

        pool_size = max(1, self._settings.embedding_onnx_pool_size)
        logger.info("正在加载 ONNX 向量模型：{}，pool_size={}", onnx_path, pool_size)
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        session_options = self._build_session_options()
        self._sessions = [
            ort.InferenceSession(
                str(onnx_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            for _ in range(pool_size)
        ]
        self._session_pool = None
        logger.info("ONNX 向量模型加载完成，sessions={}", len(self._sessions))

    def _build_session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        intra = self._settings.embedding_onnx_intra_op_threads
        if intra > 0:
            options.intra_op_num_threads = intra
        options.inter_op_num_threads = 1
        return options

    @property
    def is_loaded(self) -> bool:
        return bool(self._sessions) and self._tokenizer is not None

    @property
    def pool_size(self) -> int:
        return len(self._sessions)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.is_loaded:
            raise EncoderNotReadyError()
        session = await self._acquire_session()
        try:
            return await anyio.to_thread.run_sync(self._encode_sync, session, texts)
        finally:
            await self._release_session(session)

    async def _acquire_session(self) -> ort.InferenceSession:
        pool = await self._ensure_pool()
        return await pool.get()

    async def _release_session(self, session: ort.InferenceSession) -> None:
        pool = await self._ensure_pool()
        await pool.put(session)

    async def _ensure_pool(self) -> asyncio.Queue[ort.InferenceSession]:
        if self._session_pool is not None:
            return self._session_pool
        async with self._pool_init_lock:
            if self._session_pool is not None:
                return self._session_pool
            pool: asyncio.Queue[ort.InferenceSession] = asyncio.Queue()
            for session in self._sessions:
                pool.put_nowait(session)
            self._session_pool = pool
            return pool

    def _encode_sync(
        self,
        session: ort.InferenceSession,
        texts: list[str],
    ) -> list[list[float]]:
        assert self._tokenizer is not None
        with self._tokenize_lock:
            batch = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
        hidden = session.run(
            None,
            {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            },
        )[0]
        pooled = cls_pool(hidden)
        normalized = l2_normalize(pooled)
        return normalized.tolist()

    async def encode_one(self, text: str) -> list[float]:
        results = await self.encode([text])
        return results[0]
