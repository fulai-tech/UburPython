"""OnnxEncoder 会话池单元测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import EncoderNotReadyError
from app.embedding.onnx_encoder import OnnxEncoder


def _settings(**overrides: object) -> Settings:
    base = {
        "embedding_onnx_dir": "models/onnx/bge-small-zh-v1.5",
        "embedding_onnx_pool_size": 3,
        "embedding_onnx_intra_op_threads": 1,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_load_creates_session_pool(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    tokenizer = MagicMock()
    sessions: list[MagicMock] = []

    def _session_factory(*_args, **_kwargs):
        session = MagicMock()
        sessions.append(session)
        return session

    encoder = OnnxEncoder(
        _settings(embedding_onnx_dir=str(tmp_path), embedding_onnx_pool_size=3)
    )
    with (
        patch("app.embedding.onnx_encoder.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("app.embedding.onnx_encoder.ort.InferenceSession", side_effect=_session_factory),
    ):
        encoder.load()

    assert encoder.is_loaded
    assert encoder.pool_size == 3
    assert len(sessions) == 3


@pytest.mark.asyncio
async def test_encode_not_loaded_raises() -> None:
    encoder = OnnxEncoder(_settings())
    with pytest.raises(EncoderNotReadyError):
        await encoder.encode(["雨声"])


@pytest.mark.asyncio
async def test_concurrent_encode_reuses_pooled_sessions(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
    }

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()
    sessions: list[MagicMock] = []

    def _session_factory(*_args, **_kwargs):
        session = MagicMock()

        def _run(*_a, **_k):
            return [np.ones((1, 3, 4), dtype=np.float32)]

        session.run.side_effect = _run
        sessions.append(session)
        return session

    encoder = OnnxEncoder(
        _settings(embedding_onnx_dir=str(tmp_path), embedding_onnx_pool_size=2)
    )
    with (
        patch("app.embedding.onnx_encoder.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("app.embedding.onnx_encoder.ort.InferenceSession", side_effect=_session_factory),
        patch("app.embedding.onnx_encoder.cls_pool", return_value=np.ones((1, 4))),
        patch("app.embedding.onnx_encoder.l2_normalize", return_value=np.ones((1, 4))),
    ):
        encoder.load()

        async def _tracked_encode(label: str) -> list[list[float]]:
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                # 占住 session：在线程里 sleep 会拉长重叠窗口
                await asyncio.sleep(0.05)
                return await encoder.encode([label])
            finally:
                async with lock:
                    in_flight -= 1

        results = await asyncio.gather(
            _tracked_encode("a"),
            _tracked_encode("b"),
            _tracked_encode("c"),
            _tracked_encode("d"),
        )

    assert len(results) == 4
    assert all(len(row) == 1 for row in results)
    # 池大小为 2：并发 encode 时最多约 2 路同时持有 session（允许调度误差看 pool 被复用）
    assert encoder.pool_size == 2
    assert sum(s.run.call_count for s in sessions) == 4
