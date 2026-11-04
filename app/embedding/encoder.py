"""向量编码器工厂：本地 ONNX（zh/en int8）或 PyTorch / Qwen API 回退。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from app.es.somni_docs import resolve_search_language

if TYPE_CHECKING:
    from app.core.config import Settings


class EncoderBase(ABC):
    """Encoder / OnnxEncoder / TorchEncoder / LanguageRouterEncoder 共用接口。"""

    @abstractmethod
    def load(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    async def encode(
        self, texts: list[str], *, language: str = "zh"
    ) -> list[list[float]]: ...

    @abstractmethod
    async def encode_one(self, text: str, *, language: str = "zh") -> list[float]: ...


# 测试替身与类型标注沿用 Encoder 名称
Encoder = EncoderBase


class LanguageRouterEncoder(EncoderBase):
    """按检索语言路由到 zh / en ONNX；en 模型缺失时仅 zh 可用。"""

    def __init__(self, zh: EncoderBase, en: EncoderBase | None) -> None:
        self._zh = zh
        self._en = en

    def load(self) -> None:
        self._zh.load()
        if self._en is not None:
            self._en.load()

    @property
    def is_loaded(self) -> bool:
        if not self._zh.is_loaded:
            return False
        if self._en is None:
            return True
        return self._en.is_loaded

    def _route(self, language: str) -> EncoderBase:
        if resolve_search_language(language) == "en":
            if self._en is None:
                msg = "英文 ONNX 模型未配置或未加载，无法编码 language=en"
                raise FileNotFoundError(msg)
            return self._en
        return self._zh

    async def encode(
        self, texts: list[str], *, language: str = "zh"
    ) -> list[list[float]]:
        return await self._route(language).encode(texts, language=language)

    async def encode_one(self, text: str, *, language: str = "zh") -> list[float]:
        return await self._route(language).encode_one(text, language=language)


def create_english_encoder(settings: Settings) -> EncoderBase | None:
    """英文 ONNX。模型文件不在时返回 None。"""
    if settings.embedding_backend.lower() != "onnx":
        return None
    directory = settings.resolve_embedding_onnx_dir_en()
    if not (Path(directory) / "model.onnx").is_file():
        return None
    from app.embedding.onnx_encoder import OnnxEncoder

    return OnnxEncoder(settings.model_copy(update={"embedding_onnx_dir": directory}))


def create_encoder(settings: Settings) -> EncoderBase:
    backend = settings.embedding_backend.lower()
    if backend == "onnx":
        from app.embedding.onnx_encoder import OnnxEncoder

        zh = OnnxEncoder(settings)
        en = create_english_encoder(settings)
        return LanguageRouterEncoder(zh, en)
    if backend == "torch":
        from app.embedding.torch_encoder import TorchEncoder

        return TorchEncoder(settings)
    if backend == "qwen_api":
        from app.embedding.qwen_api_encoder import QwenApiEncoder

        return QwenApiEncoder(settings)
    msg = f"不支持的 EMBEDDING_BACKEND={settings.embedding_backend!r}，可选 onnx / torch / qwen_api"
    raise ValueError(msg)
