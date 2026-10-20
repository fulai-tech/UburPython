"""Canonical English audio tags shared by retrieval, exclusions and cache identity."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any


def _key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "_", text).strip("_")


@lru_cache(maxsize=1)
def _dictionary_asset() -> dict[str, Any]:
    asset = json.loads(Path(__file__).with_name("audio_tag_aliases.json").read_text())
    if asset["aliases"] != build_dictionary_aliases(asset["dictionary"]):
        raise ValueError("audio_tag_aliases_must_be_derived_from_dictionary")
    return asset


def build_dictionary_aliases(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Derive identities from dictionary fields; omit ambiguous display names."""
    candidates: dict[str, set[str]] = {}
    codes: dict[str, str] = {}
    for record in records:
        code = str(record.get("code") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", code):
            raise ValueError("invalid_dictionary_tag_code")
        codes[_key(code)] = code
        for field in ("code", "name", "name_en"):
            if alias := _key(record.get(field)):
                candidates.setdefault(alias, set()).add(code)
    aliases = {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}
    return {**aliases, **codes}


def dictionary_records() -> list[dict[str, Any]]:
    """Expose source rows for lexical fallback, without embedding-service dependencies."""
    return [dict(record) for record in _dictionary_asset()["dictionary"]]


@lru_cache(maxsize=1)
def tag_aliases() -> dict[str, str]:
    return _dictionary_asset()["aliases"]


# Application phase identifiers are protocol values, not invented material tags.
_PHASE_ALIASES = {
    "u": "unwind",
    "s": "soothe",
    "g": "guard",
    "w": "wake",
    "w_s1": "wake",
    "w_s2": "wake",
    "w_s3": "wake",
}


def canonical_tag(value: Any, *, strict: bool = True) -> str:
    """Resolve audited aliases, accepting forward-compatible English snake-case codes."""
    raw = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = _key(raw)
    if not text:
        if raw and strict:
            raise ValueError("invalid_audio_tag: empty after normalization")
        return ""
    code = _PHASE_ALIASES.get(text) or tag_aliases().get(text)
    if code:
        return code
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", raw):
        return raw.casefold()
    if strict:
        raise ValueError("unknown_audio_tag: use a reviewed English dictionary code")
    return ""


def canonical_tags(values: Sequence[Any], *, strict: bool = True) -> list[str]:
    if strict and any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("invalid_audio_tag: expected nonempty strings")
    return list(
        dict.fromkeys(code for value in values if (code := canonical_tag(value, strict=strict)))
    )


def reference_tag(item: Mapping[str, Any]) -> str:
    """Prefer dictionary code, then a legacy display alias; never infer from title prose."""
    code = str(item.get("code") or "")
    if resolved := canonical_tag(code, strict=False):
        return resolved
    for field in ("name", "name_en", "en_name", "label"):
        value = item.get(field)
        if value and _key(value) in tag_aliases():
            return canonical_tag(value)
    return canonical_tag(code, strict=False)


def material_tag_codes(material: Mapping[str, Any]) -> set[str]:
    """Read real structured refs and legacy flattened labels into English identities."""
    result: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, Mapping):
            code = reference_tag(value)
            if code:
                result.add(code)
            parent = value.get("parent_tag_code") or value.get("parent_tag_name")
            if parent and (code := canonical_tag(parent, strict=False)):
                result.add(code)
            if not any(key in value for key in ("code", "name", "label")):
                for child in value.values():
                    add(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                add(child)
        elif value is not None:
            text = str(value)
            if ":" in text:
                text = text.split(":", 1)[1]
            if code := canonical_tag(text, strict=False):
                result.add(code)

    for field in (
        "sleep_stage_tags",
        "content_form_tags",
        "mechanism_tags",
        "medical_risk_tags",
        "content_tag_codes",
        "content_tags",
        "tags",
        "tag",
        "content_format",
        "multi_sound_label",
        "mixed_sound_labels",
    ):
        add(material.get(field))
    return result
