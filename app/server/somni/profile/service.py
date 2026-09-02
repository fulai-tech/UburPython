"""量产用户画像：按 type 查询长期/短期画像。"""

from __future__ import annotations

from typing import Any, Literal

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.bson_util import bson_to_jsonable
from app.core.codes import HttpStatus
from app.core.config import Settings
from app.core.exceptions import AppError
from app.server.somni.profile.store import ProfileStore

ProfileType = Literal["long_terms", "short_terms", "both"]


class ProfileService:
    def __init__(
        self,
        client: AsyncIOMotorClient | None,
        settings: Settings,
        store: ProfileStore | None = None,
    ) -> None:
        self._store = store or ProfileStore(client, settings)

    async def get_user_profile(
        self,
        uid: str,
        profile_type: str,
        record_date: str = "",
    ) -> dict[str, Any]:
        normalized = _normalize_profile_type(profile_type)
        if normalized == "both":
            return {"profile": await self._both(uid, record_date)}
        if normalized == "short_terms":
            return {
                "profile": await self._short_term(uid, record_date),
            }
        return {"profile": await self._long_term(uid)}

    async def _long_term(self, uid: str) -> dict[str, Any]:
        doc = await self._store.find_long_term_profile(uid)
        if doc is None:
            raise AppError(
                message=f"画像不存在：{uid}",
                status_code=HttpStatus.NOT_FOUND,
            )
        return _profile_field(doc, "long_term_profile")

    async def _short_term(self, uid: str, record_date: str) -> dict[str, Any]:
        day = _require_record_date(record_date, "short_terms")
        doc = await self._store.find_short_term_profile(uid, day)
        if doc is None:
            raise AppError(
                message=f"画像不存在：{uid}/{day}",
                status_code=HttpStatus.NOT_FOUND,
            )
        return _profile_field(doc, "short_term_profile")

    async def _both(self, uid: str, record_date: str) -> dict[str, Any]:
        day = _require_record_date(record_date, "both")
        long_doc = await self._store.find_long_term_profile(uid)
        short_doc = await self._store.find_short_term_profile(uid, day)
        missing: list[str] = []
        if long_doc is None:
            missing.append("long_terms")
        if short_doc is None:
            missing.append("short_terms")
        if missing:
            raise AppError(
                message=f"画像不存在：{uid}/{day}（{'+'.join(missing)}）",
                status_code=HttpStatus.NOT_FOUND,
            )
        return {
            "long_term_profile": _profile_field(long_doc, "long_term_profile"),
            "short_term_profile": _profile_field(short_doc, "short_term_profile"),
        }


def _require_record_date(record_date: str, profile_type: str) -> str:
    day = record_date.strip()
    if not day:
        raise AppError(
            message=f"{profile_type} 类型必须传递 record_date",
            status_code=HttpStatus.BAD_REQUEST,
        )
    return day


def _profile_field(doc: dict[str, Any], field: str) -> dict[str, Any]:
    raw = bson_to_jsonable(doc)
    profile = raw.get(field)
    return profile if isinstance(profile, dict) else {}


def _normalize_profile_type(profile_type: str) -> ProfileType:
    normalized = profile_type.strip().lower()
    if normalized not in ("long_terms", "short_terms", "both"):
        raise AppError(
            message="type 必须是 long_terms、short_terms 或 both",
            status_code=HttpStatus.BAD_REQUEST,
        )
    return normalized  # type: ignore[return-value]
