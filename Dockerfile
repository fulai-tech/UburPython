# exporter：一次性导出 ONNX（含 torch，不进入最终镜像）
FROM python:3.12-slim-bookworm AS exporter

ARG APT_MIRROR=mirrors.tencent.com
ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

WORKDIR /export
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources; \
    else \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" "sentence-transformers>=3.3.0" torch onnxscript

COPY scripts/export_onnx_model.py ./scripts/
ENV CUDA_VISIBLE_DEVICES="" 
RUN python scripts/export_onnx_model.py --output-dir /export/onnx

# builder：安装生产依赖（onnxruntime，无 PyTorch）
FROM python:3.12-slim-bookworm AS builder

ARG APT_MIRROR=mirrors.tencent.com
ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

WORKDIR /app

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources; \
    else \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" .

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=exporter /export/onnx ./models/onnx/bge-small-zh-v1.5

COPY scripts/__init__.py scripts/sync_es_from_comm.py ./scripts/

ENV HF_HOME=/data/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/data/huggingface \
    EMBEDDING_BACKEND=onnx \
    EMBEDDING_ONNX_DIR=models/onnx/bge-small-zh-v1.5

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
