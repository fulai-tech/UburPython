"""ReportService.get_profile。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError
from app.server.somni.report.service import ReportService


@pytest.mark.asyncio
async def test_get_profile_returns_profile_text() -> None:
    store = MagicMock()
    store.find_user_profile = AsyncMock(
        return_value={"profile_text": "用户基础信息：性别女"}
    )
    svc = ReportService(client=None, settings=Settings(), store=store)
    assert await svc.get_profile("u1", "2026-08-30") == {
        "profile_text": "用户基础信息：性别女"
    }
    store.find_user_profile.assert_awaited_once_with("u1", "2026-08-30")


@pytest.mark.asyncio
async def test_get_profile_missing_field_returns_empty_string() -> None:
    store = MagicMock()
    store.find_user_profile = AsyncMock(
        return_value={"uid": "u1", "record_date": "2026-08-30"}
    )
    svc = ReportService(client=None, settings=Settings(), store=store)
    assert await svc.get_profile("u1", "2026-08-30") == {"profile_text": ""}


@pytest.mark.asyncio
async def test_get_profile_not_found() -> None:
    store = MagicMock()
    store.find_user_profile = AsyncMock(return_value=None)
    svc = ReportService(client=None, settings=Settings(), store=store)
    with pytest.raises(AppError) as exc_info:
        await svc.get_profile("u1", "2026-08-30")
    assert exc_info.value.status_code == HttpStatus.NOT_FOUND
    assert "画像不存在" in exc_info.value.message
