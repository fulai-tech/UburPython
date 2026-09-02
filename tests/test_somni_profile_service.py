"""ProfileService.get_user_profile。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.codes import HttpStatus
from app.core.exceptions import AppError
from app.server.somni.profile.service import ProfileService


@pytest.mark.asyncio
async def test_get_long_term_profile() -> None:
    store = MagicMock()
    store.find_long_term_profile = AsyncMock(
        return_value={
            "uid": "u1",
            "status": "valid",
            "long_term_profile": {"sleep_type": "sleep_onset_insomnia"},
        }
    )
    svc = ProfileService(None, MagicMock(), store=store)
    payload = await svc.get_user_profile("u1", "long_terms")
    assert payload["profile"]["sleep_type"] == "sleep_onset_insomnia"
    assert "uid" not in payload["profile"]
    store.find_long_term_profile.assert_awaited_once_with("u1")


@pytest.mark.asyncio
async def test_get_short_term_profile() -> None:
    store = MagicMock()
    store.find_short_term_profile = AsyncMock(
        return_value={
            "uid": "u1",
            "record_date": "2026-09-01",
            "status": "active",
            "short_term_profile": {
                "morning_feedback": {"subjective_sleep_quality": 6}
            },
        }
    )
    svc = ProfileService(None, MagicMock(), store=store)
    payload = await svc.get_user_profile("u1", "short_terms", "2026-09-01")
    assert payload["profile"]["morning_feedback"]["subjective_sleep_quality"] == 6
    assert "record_date" not in payload["profile"]
    store.find_short_term_profile.assert_awaited_once_with("u1", "2026-09-01")


@pytest.mark.asyncio
async def test_short_terms_requires_record_date() -> None:
    svc = ProfileService(None, MagicMock(), store=MagicMock())
    with pytest.raises(AppError) as exc:
        await svc.get_user_profile("u1", "short_terms")
    assert exc.value.status_code == HttpStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_invalid_type() -> None:
    svc = ProfileService(None, MagicMock(), store=MagicMock())
    with pytest.raises(AppError) as exc:
        await svc.get_user_profile("u1", "daily")
    assert exc.value.status_code == HttpStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_not_found() -> None:
    store = MagicMock()
    store.find_long_term_profile = AsyncMock(return_value=None)
    svc = ProfileService(None, MagicMock(), store=store)
    with pytest.raises(AppError) as exc:
        await svc.get_user_profile("u1", "long_terms")
    assert exc.value.status_code == HttpStatus.NOT_FOUND
