from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from functools import wraps
from inspect import signature
from typing import Any


def parse_offset(value: str) -> timezone:
    if not re.fullmatch(r"[+-]\d{2}:\d{2}", value):
        raise ValueError("invalid_timezone_offset")
    hours, minutes = map(int, value[1:].split(":"))
    offset = (hours * 60 + minutes) * (-1 if value[0] == "-" else 1)
    if minutes >= 60 or not -720 <= offset <= 840:
        raise ValueError("invalid_timezone_offset")
    return timezone(timedelta(minutes=offset))


def parse_date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("invalid_local_date")
    return date.fromisoformat(value)


def instant(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return None
    return value.astimezone(UTC)


@dataclass(frozen=True)
class TimeScope:
    end: date
    offset: str

    def start(self, days: int) -> date:
        return self.end - timedelta(days=days)

    def bounds(self, days: int) -> tuple[datetime, datetime]:
        zone = parse_offset(self.offset)
        return tuple(
            datetime.combine(day, datetime.min.time(), zone).astimezone(UTC)
            for day in (self.start(days), self.end)
        )


def resolve_scope(end: str, offset: str) -> TimeScope:
    parse_offset(offset)
    return TimeScope(parse_date(end), offset)


def attribute_sleep_date(end: datetime, offset: str) -> str:
    if end.tzinfo is None:
        raise ValueError("timezone_required")
    return end.astimezone(parse_offset(offset)).date().isoformat()


# Request-local policy: generation reads use current versions; evidence pagination
# and reminders retain their independent read-time semantics.

_current_generation_read: ContextVar[bool] = ContextVar("current_generation_read", default=False)
_business_instant: ContextVar[datetime | None] = ContextVar(
    "generation_business_instant", default=None
)


def current_generation_read() -> bool:
    return _current_generation_read.get()


def visible_by(value: datetime, read_at: datetime) -> bool:
    return current_generation_read() or value <= read_at


def observation_reference(read_at: datetime) -> datetime:
    return _business_instant.get() or read_at


def without_legacy_cutoff(value: Any) -> Any:
    """Ignore only retired visibility metadata, including nested stored records."""
    if isinstance(value, dict):
        return {
            key: without_legacy_cutoff(item)
            for key, item in value.items()
            if key not in {"data_cutoff_at", "data_cutoff_at_utc", "cutoff_policy_version"}
        }
    if isinstance(value, (list, tuple)):
        return [without_legacy_cutoff(item) for item in value]
    return value


def current_generation_inputs[**P, R](
    method: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Scope current-version selection to one async profile/plan request."""
    parameters = signature(method)

    @wraps(method)
    async def wrapped(*args: Any, **kwargs: Any) -> R:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"data_cutoff_at", "data_cutoff_at_utc", "cutoff_policy_version"}
        }
        bound = parameters.bind(*args, **kwargs).arguments
        day = bound.get("target_local_date")
        if not day and bound.get("review_end_local_date"):
            day = (parse_date(bound["review_end_local_date"]) - timedelta(days=1)).isoformat()
        business = None
        if day:
            business = datetime.combine(
                parse_date(day),
                datetime.min.time(),
                parse_offset(bound.get("timezone_offset") or "+00:00"),
            )
        token = _current_generation_read.set(True)
        business_token = _business_instant.set(business)
        try:
            return without_legacy_cutoff(await method(*args, **kwargs))
        finally:
            _business_instant.reset(business_token)
            _current_generation_read.reset(token)

    return wrapped
