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


# Questionnaire answer semantics, deliberately separate from dictionary aliases:
# "无" is a valid engineering value but means no exclusion in a dislike field.
_NO_DISLIKE_ANSWERS = frozenset(
    {
        "无",
        "没有",
        "无厌恶",
        "没有不喜欢的声音",
        "无不喜欢的声音",
        "none",
        "no dislikes",
        "no_dislikes",
        "no preference",
        "nothing",
    }
)


def is_no_dislike_answer(value: Any) -> bool:
    """Recognize an exact questionnaire no-exclusion answer, without substring guesses."""
    return (
        isinstance(value, str)
        and unicodedata.normalize("NFKC", value).strip().casefold() in _NO_DISLIKE_ANSWERS
    )


def canonical_disliked_tags(values: Sequence[Any], *, strict: bool = True) -> list[str]:
    """Return reviewed exclusions; unmatched free text never disables retrieval."""
    resolved = resolve_audio_preferences(disliked=values)
    return resolved["disliked_codes"]


_PREFERENCE_TYPES = frozenset({"content_form", "mechanism", "audio_engineering"})
_CONTENT_TYPES = frozenset({"content_form", "mechanism"})


@lru_cache(maxsize=1)
def _preference_rules() -> dict[str, Any]:
    rules = json.loads((Path(__file__).with_name("audio_preference_rules.json")).read_text())
    validate_audio_tag_contract(dictionary_records(), rules)
    return rules


@lru_cache(maxsize=1)
def audio_tag_contract_version() -> str:
    """Bind caches and preference evidence to the exact dictionary and mapping."""
    import hashlib

    payload = {"dictionary": dictionary_records(), "rules": _preference_rules(), "resolver": 2}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validated_audio_tags(
    values: Sequence[Any], *, allowed_types: frozenset[str] | None = None
) -> list[str]:
    """Reject invented, disabled or wrong-dimension machine tags before retrieval."""
    known: dict[str, list[dict[str, Any]]] = {}
    for row in dictionary_records():
        if row.get("status") == "启用":
            known.setdefault(row["code"], []).append(row)
    codes = canonical_tags(values)
    for code in codes:
        matches = [
            row
            for row in known.get(code, [])
            if allowed_types is None or row["type"] in allowed_types
        ]
        if not matches or any(
            row["type"] == "audio_engineering"
            and row.get("parent_tag_id")
            and len({x.get("parent_tag_id") for x in matches}) > 1
            for row in matches
        ):
            raise ValueError("unknown_audio_tag: code must be enabled and dimension unambiguous")
    return codes


def _descendant_codes(codes: Sequence[str]) -> list[str]:
    rows = dictionary_records()
    children: dict[str, list[str]] = {}
    for row in rows:
        if row.get("status") == "启用" and row.get("type") == "content_form":
            children.setdefault(str(row.get("parent_tag_id") or ""), []).append(row["code"])
    by_code = {
        row["code"]: str(row.get("id") or "") for row in rows if row.get("type") == "content_form"
    }
    expanded = set(codes)
    pending = list(codes)
    while pending:
        code = pending.pop()
        for child in children.get(by_code.get(code, ""), []) if by_code.get(code) else []:
            if child not in expanded:
                expanded.add(child)
                pending.append(child)
    return sorted(expanded)


def _preference_term_codes(value: str, *, dislike: bool) -> list[str]:
    text = unicodedata.normalize("NFKC", value).strip()
    # Match reviewed whole terms, optionally with a conventional negative prefix.
    text, negative_prefix_count = re.subn(
        r"^(?:(?:我)?(?:(?:不喜欢|讨厌|不爱)(?:听)?|不想听|不想要|不要(?:放|听)?|避免|排除|拒绝|别放|别)|"
        r"(?:i\s+)?(?:do\s+not\s+like|don['’]t\s+like|do\s+not\s+want|don['’]t\s+want|"
        r"dislike|hate|avoid|exclude|without|no|not)\s+)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if negative_prefix_count and not dislike:
        return []
    try:
        return validated_audio_tags(
            [text], allowed_types=_PREFERENCE_TYPES if dislike else _CONTENT_TYPES
        )
    except ValueError:
        pass
    for group in _preference_rules()["groups"]:
        if _key(text) in {_key(alias) for alias in group["aliases"]}:
            return list(group["codes"])
    return []


def resolve_audio_preferences(
    *, preferred: Sequence[Any] = (), disliked: Sequence[Any] = (), blocked: Sequence[Any] = ()
) -> dict[str, Any]:
    """Resolve reviewed terms; unknown free text is diagnostic only, never a silence choice."""
    rules = _preference_rules()
    result: dict[str, Any] = {
        "preferred_codes": [],
        "disliked_codes": [],
        "unresolved_preferred": [],
        "unresolved_disliked": [],
        "sound_disabled": False,
        "version": audio_tag_contract_version(),
    }
    # Programmatic safety constraints remain strict; only free text may be ignored.
    result["disliked_codes"].extend(
        _descendant_codes(validated_audio_tags(blocked, allowed_types=_PREFERENCE_TYPES))
    )
    for dislike, values in ((False, preferred), (True, disliked)):
        for value in values:
            if not isinstance(value, str) or not value.strip():
                result["unresolved_disliked" if dislike else "unresolved_preferred"].append(
                    str(value)
                )
                continue
            if dislike and is_no_dislike_answer(value):
                continue
            if not dislike and _key(value) in {_key(x) for x in rules["silence_answers"]}:
                result["sound_disabled"] = True
                continue
            if not dislike and _key(value) in {_key(x) for x in rules["no_preference_answers"]}:
                continue
            # Resolve the complete label first (e.g. ASMR/耳语) before splitting lists.
            whole_codes = _preference_term_codes(value, dislike=dislike)
            if whole_codes:
                result["disliked_codes" if dislike else "preferred_codes"].extend(
                    _descendant_codes(whole_codes)
                )
                continue
            parts = re.split(
                r"[,，、;；/\n&]|\s+and\s+|和|以及|与",
                unicodedata.normalize("NFKC", value),
                flags=re.IGNORECASE,
            )
            for part in parts:
                if not part.strip():
                    continue
                if dislike and is_no_dislike_answer(part.strip()):
                    continue
                if not dislike and _key(part) in {_key(x) for x in rules["silence_answers"]}:
                    result["sound_disabled"] = True
                    continue
                if not dislike and _key(part) in {_key(x) for x in rules["no_preference_answers"]}:
                    continue
                codes = _preference_term_codes(part, dislike=dislike)
                key = "disliked_codes" if dislike else "preferred_codes"
                if codes:
                    result[key].extend(_descendant_codes(codes))
                else:
                    result["unresolved_disliked" if dislike else "unresolved_preferred"].append(
                        part.strip()
                    )
    for key in ("preferred_codes", "disliked_codes", "unresolved_preferred", "unresolved_disliked"):
        result[key] = sorted(set(result[key]))
    result["preferred_codes"] = sorted(
        set(result["preferred_codes"]) - set(result["disliked_codes"])
    )
    return result


def preference_audio_tags(values: Sequence[Any], *, dislike: bool = False) -> list[str]:
    """Translate reviewed legacy terms, leaving unknown free text out of machine tags."""
    resolved = resolve_audio_preferences(**{"disliked" if dislike else "preferred": values})
    return resolved["disliked_codes" if dislike else "preferred_codes"]


def validate_audio_tag_contract(
    records: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
) -> None:
    """Reject drift, ambiguous aliases and broken hierarchy before using an asset."""
    by_code = {row["code"]: row for row in records}
    identities = {
        (row["type"], str(row.get("parent_tag_id") or ""), row["code"]) for row in records
    }
    if len(identities) != len(records):
        raise ValueError("duplicate_audio_tag_identity")
    ids = {str(row.get("id") or ""): row for row in records if row.get("id")}
    for row in records:
        parent = str(row.get("parent_tag_id") or "")
        visited = {str(row.get("id") or "")}
        while parent:
            if parent not in ids or parent in visited:
                raise ValueError("invalid_audio_tag_parent")
            visited.add(parent)
            parent_row = ids[parent]
            if parent_row["type"] != row["type"]:
                raise ValueError("invalid_audio_tag_parent_type")
            parent = str(parent_row.get("parent_tag_id") or "")
    aliases: dict[str, set[str]] = {}
    official = build_dictionary_aliases(records)
    for group in rules["groups"]:
        codes = set(group["codes"])
        if not codes or any(
            code not in by_code
            or by_code[code].get("status") != "启用"
            or by_code[code]["type"] not in _CONTENT_TYPES
            for code in codes
        ):
            raise ValueError("invalid_audio_preference_group")
        for alias in group["aliases"]:
            key = _key(alias)
            if (key in aliases and aliases[key] != codes) or (
                key in official and official[key] not in codes
            ):
                raise ValueError("ambiguous_audio_preference_alias")
            aliases[key] = codes


def material_is_excluded(material: Mapping[str, Any], disliked: Sequence[Any]) -> bool:
    """Recheck fresh material metadata after retrieval/cache lookup, including voice values."""
    excluded = set(canonical_disliked_tags(disliked))
    if (excluded - {"voice"}) & material_tag_codes(material):
        return True
    if "voice" in excluded:
        values = []
        for row in material.get("audio_engineering_tags") or []:
            if isinstance(row, Mapping) and reference_tag(row) in {"voice", "voice_position"}:
                value = row.get("value")
                values.append(value.get("code") if isinstance(value, Mapping) else None)
        if not values or any(value != "none" for value in values):
            return True
    return False
