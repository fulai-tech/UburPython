from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any

from app.server.somni.facts.time import without_legacy_cutoff


class DataError(Exception):
    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def section(value: Any = None, status: str = "ok", reasons: list[str] | None = None) -> dict:
    result = {"status": status, "missing_reasons": sorted(set(reasons or []))}
    if value is not None:
        result["value"] = value
    return result


def canonical_hash(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, float) and math.isfinite(item) and item.is_integer():
            return int(item)
        return item

    try:
        encoded = json.dumps(
            normalize(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (ValueError, TypeError) as exc:
        raise DataError("INVALID_ARGUMENT", "invalid_canonical_content") from exc
    return hashlib.sha256(encoded).hexdigest()


PROFILE_FIELDS = (
    "uid",
    "schema_version",
    "profile_snapshot_id",
    "snapshot_revision",
    "record_date",
    "generated_at_utc",
    "status",
    "long_term_content_hash",
    "long_term_profile",
    "short_term_profile",
    "rolling_baseline",
    "profile_quality",
    "based_on",
)


def compute_long_term_content_hash(body: dict, schema_version: str) -> str:
    if schema_version == "0.0.1":
        try:
            encoded = json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (ValueError, TypeError) as exc:
            raise DataError("INVALID_ARGUMENT", "invalid_canonical_content") from exc
        return "sha256:" + hashlib.sha256(encoded).hexdigest()
    return canonical_hash(body)


def validate_long_term_profile(profile: dict) -> None:
    body = profile.get("long_term_profile")
    if not isinstance(body, dict):
        raise DataError("INVALID_ARGUMENT", "incomplete_profile_content")
    try:
        json.dumps(body, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise DataError("INVALID_ARGUMENT", "invalid_canonical_content") from exc


def validate_profile(profile: dict, uid: str, *, verify_long_term_hash: bool = False) -> dict:
    profile = without_legacy_cutoff(profile)
    if profile.get("uid") != uid:
        raise DataError("PERMISSION_DENIED", "profile_owner_mismatch")
    required = ("schema_version", "record_date", "generated_at_utc", "status")
    if any(not profile.get(key) for key in required):
        raise DataError("INVALID_ARGUMENT", "incomplete_profile_metadata")
    for key in (
        "long_term_profile",
        "short_term_profile",
        "rolling_baseline",
        "profile_quality",
        "based_on",
    ):
        if not isinstance(profile.get(key), dict):
            raise DataError("INVALID_ARGUMENT", "incomplete_profile_content")
    content = {key: profile[key] for key in PROFILE_FIELDS if key in profile}
    generated_hash = canonical_hash(content)
    return {**content, "content_hash": profile.get("content_hash", generated_hash)}
