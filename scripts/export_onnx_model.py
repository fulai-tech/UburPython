#!/usr/bin/env python3
"""导出 bge-small-zh-v1.5 为 ONNX（开发环境，需 pip install -e '.[dev]'）。

仅导出 Transformer 输出 last_hidden_state；mean pool + L2 归一化在 OnnxEncoder 内完成，
与 sentence-transformers 数值路径一致。

用法:
  .venv/bin/python scripts/export_onnx_model.py
  .venv/bin/python scripts/export_onnx_model.py --output-dir models/onnx/bge-small-zh-v1.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 国内服务器构建时无法直连 huggingface.co，默认走 HF 镜像站
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel


class _TransformerExport(nn.Module):
    """只导出 last_hidden_state，池化留给运行时 Python 实现。"""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def export_model(model_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    # SDPA 在 ONNX trace 下数值偏差大，必须用 eager attention
    auto_model = AutoModel.from_pretrained(model_id, attn_implementation="eager")
    auto_model.eval()
    st_model = SentenceTransformer(model_id, device="cpu")

    wrapper = _TransformerExport(auto_model).to("cpu")
    sample = st_model.tokenizer(
        ["测试文本", "雨声森林白噪音"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=18,
        dynamo=False,
    )
    auto_model.config.save_pretrained(output_dir)
    st_model.tokenizer.save_pretrained(output_dir)
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="BAAI/bge-small-zh-v1.5",
        help="HuggingFace 模型 ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "models/onnx/bge-small-zh-v1.5",
        help="ONNX 与 tokenizer 输出目录",
    )
    args = parser.parse_args()
    onnx_path = export_model(args.model_id, args.output_dir)
    print(f"已导出 ONNX：{onnx_path}")
    print(f"Tokenizer 目录：{args.output_dir}")


if __name__ == "__main__":
    main()
