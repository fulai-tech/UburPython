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
def tag_aliases() -> dict[str, str]:
    return json.loads(Path(__file__).with_name("audio_tag_aliases.json").read_text())["aliases"]


def canonical_tag(value: Any, *, strict: bool = True) -> str:
    """Resolve audited aliases, accepting forward-compatible English snake-case codes."""
    text = _key(value)
    if not text:
        return ""
    code = tag_aliases().get(text)
    if code:
        return code
    if re.fullmatch(r"[a-z][a-z0-9_]*", text):
        return text
    if strict:
        raise ValueError("unknown_audio_tag: use a reviewed English dictionary code")
    return ""


def canonical_tags(values: Sequence[Any], *, strict: bool = True) -> list[str]:
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
