"""手板 / 量产 gRPC bootstrap。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.main import AppState
from app.server.bootstrap import start_grpc_servers, stop_grpc_servers


@pytest.mark.asyncio
async def test_start_both_servers_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_hb = MagicMock()
    fake_hb.add_insecure_port = MagicMock(return_value=50051)
    fake_hb.start = AsyncMock()
    fake_sm = MagicMock()
    fake_sm.add_insecure_port = MagicMock(return_value=50052)
    fake_sm.start = AsyncMock()
    servers = iter([fake_hb, fake_sm])
    monkeypatch.setattr(
        "app.server.bootstrap.grpc.aio.server",
        lambda: next(servers),
    )
    settings = Settings(grpc_enabled=True, somni_grpc_enabled=True)
    state = AppState(settings=settings)
    result = await start_grpc_servers(state, settings)
    assert result.handboard is fake_hb
    assert result.somni is fake_sm
    fake_hb.start.assert_awaited_once()
    fake_sm.start.assert_awaited_once()
    fake_hb.stop = AsyncMock()
    fake_sm.stop = AsyncMock()
    await stop_grpc_servers(result)
    fake_hb.stop.assert_awaited_once()
    fake_sm.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_skips_servers() -> None:
    settings = Settings(grpc_enabled=False, somni_grpc_enabled=False)
    state = AppState(settings=settings)
    result = await start_grpc_servers(state, settings)
    assert result.handboard is None
    assert result.somni is None
