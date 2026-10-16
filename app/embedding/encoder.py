"""向量编码器工厂：按配置选择 ONNX（生产）或 PyTorch（对比/回退）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings


class EncoderBase(ABC):
    """Encoder / OnnxEncoder / TorchEncoder 共用接口。"""

    @abstractmethod
    def load(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    async def encode(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def encode_one(self, text: str) -> list[float]: ...


# 测试替身与类型标注沿用 Encoder 名称
Encoder = EncoderBase


def create_english_encoder(settings: Settings) -> EncoderBase | None:
    """英文标签名向量。模型文件不在时返回 None，中文路径不受影响。"""
    if settings.embedding_backend.lower() != "onnx":
        return None
    from pathlib import Path

    directory = settings.embedding_onnx_dir_en.strip()
    if not directory:
        directory = str(Path(settings.embedding_onnx_dir).parent / "bge-small-en-v1.5-int8")
    if not (Path(directory) / "model.onnx").is_file():
        return None
    from app.embedding.onnx_encoder import OnnxEncoder

    return OnnxEncoder(settings.model_copy(update={"embedding_onnx_dir": directory}))


def create_encoder(settings: Settings) -> EncoderBase:
    backend = settings.embedding_backend.lower()
    if backend == "onnx":
        from app.embedding.onnx_encoder import OnnxEncoder

        return OnnxEncoder(settings)
    if backend == "torch":
        from app.embedding.torch_encoder import TorchEncoder

        return TorchEncoder(settings)
    if backend == "qwen_api":
        from app.embedding.qwen_api_encoder import QwenApiEncoder

        return QwenApiEncoder(settings)
    msg = f"不支持的 EMBEDDING_BACKEND={settings.embedding_backend!r}，可选 onnx / torch / qwen_api"
    raise ValueError(msg)
