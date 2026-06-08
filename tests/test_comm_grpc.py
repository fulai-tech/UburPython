"""comm-service gRPC 集成探测（默认跳过，需显式开启）。

运行:
  COMM_GRPC_INTEGRATION=1 pytest tests/test_comm_grpc.py -v
"""

from __future__ import annotations

import os

import pytest

from app.bionode_grpc_clients import CommClient
from app.core.config import get_settings


@pytest.mark.skipif(
    os.getenv("COMM_GRPC_INTEGRATION") != "1",
    reason="设置 COMM_GRPC_INTEGRATION=1 才连真实 comm-service",
)
@pytest.mark.asyncio
async def test_comm_grpc_ping() -> None:
    settings = get_settings()
    client = CommClient(settings)
    await client.connect()
    try:
        tag_count = await client.ping(timeout_sec=15.0)
    finally:
        await client.close()
    assert tag_count >= 0
