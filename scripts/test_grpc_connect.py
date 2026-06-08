#!/usr/bin/env python3
"""探测 comm-service gRPC 连通性（读取项目根 .env 的 COMM_GRPC_*）。

用法（须先激活虚拟环境或直接用 .venv 解释器）:
  source .venv/bin/activate && python scripts/test_grpc_connect.py
  .venv/bin/python scripts/test_grpc_connect.py --timeout 15
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 允许直接 python scripts/test_grpc_connect.py（无需手动 PYTHONPATH）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("test_grpc_connect")


def _ensure_project_deps() -> None:
    try:
        import grpc  # noqa: F401
    except ImportError:
        _log.error("未找到项目依赖，请先: source .venv/bin/activate && pip install -e .")
        raise SystemExit(3)


async def run_probe(timeout_sec: float) -> int:
    from app.bionode_grpc_clients import CommClient
    from app.core.config import get_settings

    settings = get_settings()
    target = settings.comm_grpc_target
    _log.info(
        "comm-service 目标：%s（TLS=%s）",
        target,
        settings.comm_grpc_use_tls,
    )

    client = CommClient(settings)
    try:
        await client.connect()
        tag_count = await client.ping(timeout_sec=timeout_sec)
    except asyncio.TimeoutError:
        _log.error("gRPC 通道在 %ss 内未就绪（网络不可达或端口未放行）", timeout_sec)
        return 2
    except Exception as exc:
        _log.error("gRPC 调用失败：%s", exc)
        return 1
    finally:
        await client.close()

    _log.info("连通成功，GetDistinctTags 返回 %s 个标签", tag_count)
    return 0


def main() -> None:
    _ensure_project_deps()
    parser = argparse.ArgumentParser(description="探测 comm-service gRPC")
    parser.add_argument("--timeout", type=float, default=10.0, help="通道就绪与 RPC 超时（秒）")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_probe(args.timeout)))


if __name__ == "__main__":
    main()
