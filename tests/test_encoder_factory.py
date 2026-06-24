"""Encoder 工厂与后端选择单元测试。"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.embedding.encoder import create_encoder
from app.embedding.onnx_encoder import OnnxEncoder
from app.embedding.qwen_api_encoder import QwenApiEncoder
from app.embedding.torch_encoder import TorchEncoder


def test_create_encoder_returns_onnx_by_default() -> None:
    encoder = create_encoder(Settings(embedding_backend="onnx"))
    assert isinstance(encoder, OnnxEncoder)


def test_create_encoder_returns_torch_when_configured() -> None:
    encoder = create_encoder(Settings(embedding_backend="torch"))
    assert isinstance(encoder, TorchEncoder)


def test_create_encoder_returns_qwen_api_when_configured() -> None:
    encoder = create_encoder(Settings(embedding_backend="qwen_api"))
    assert isinstance(encoder, QwenApiEncoder)


def test_create_encoder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="不支持的 EMBEDDING_BACKEND"):
        create_encoder(Settings(embedding_backend="unknown"))
