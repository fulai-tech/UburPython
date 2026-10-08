from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone


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
