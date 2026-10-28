# exporter：一次性导出 ONNX（含 torch，不进入最终镜像）
FROM python:3.12-slim-bookworm AS exporter

ARG APT_MIRROR=mirrors.tencent.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG HF_ENDPOINT=https://hf-mirror.com

WORKDIR /export
ENV HF_ENDPOINT=${HF_ENDPOINT} \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    CUDA_VISIBLE_DEVICES=""
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

# 只走清华源。单次下载被截断时再试，不再使用腾讯 PyPI。
RUN set -eu; \
    pip_install() { \
      host=$$(printf '%s\n' "${PIP_INDEX_URL}" | sed -E 's#^[a-z]+://([^/:]+).*#\1#'); \
      n=0; \
      while [ "$$n" -lt 3 ]; do \
        n=$$((n + 1)); \
        echo "pip install via ${PIP_INDEX_URL} attempt $$n"; \
        if pip install --no-cache-dir --timeout 120 --retries 3 \
            --trusted-host "$$host" -i "${PIP_INDEX_URL}" "$$@"; then \
          return 0; \
        fi; \
        echo "pip install failed, attempt $$n" >&2; \
        sleep $$((n * 5)); \
      done; \
      return 1; \
    }; \
    pip_install --upgrade pip; \
    pip_install "sentence-transformers>=3.3.0" torch onnxscript

COPY scripts/export_onnx_model.py ./scripts/
RUN python scripts/export_onnx_model.py --output-dir /export/onnx

# builder：安装生产依赖（onnxruntime，无 PyTorch）
FROM python:3.12-slim-bookworm AS builder

ARG APT_MIRROR=mirrors.tencent.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

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

# 只走清华源。单次下载被截断时再试，不再使用腾讯 PyPI。
RUN set -eu; \
    pip_install() { \
      host=$$(printf '%s\n' "${PIP_INDEX_URL}" | sed -E 's#^[a-z]+://([^/:]+).*#\1#'); \
      n=0; \
      while [ "$$n" -lt 3 ]; do \
        n=$$((n + 1)); \
        echo "pip install via ${PIP_INDEX_URL} attempt $$n"; \
        if pip install --no-cache-dir --timeout 120 --retries 3 \
            --trusted-host "$$host" -i "${PIP_INDEX_URL}" "$$@"; then \
          return 0; \
        fi; \
        echo "pip install failed, attempt $$n" >&2; \
        sleep $$((n * 5)); \
      done; \
      return 1; \
    }; \
    pip_install --upgrade pip; \
    pip_install .

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
