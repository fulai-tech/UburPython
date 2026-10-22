"""Bounded profile provenance; never use this projection before policy evaluation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

PROVENANCE_VERSION = "profile_provenance.v1"
MAX_SOURCE_REFS = 256
MAX_PROVENANCE_BYTES = 128 * 1024
_REF_KEYS = {"source_refs", "source_manifest"}


def _json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


def _digest(value: Any) -> str:
    def stable(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: stable(child) for key, child in item.items() if key != "observed_at"}
        if isinstance(item, list):
            return sorted([stable(child) for child in item], key=_json)
        return item

    return "sha256:" + hashlib.sha256(_json(stable(value))).hexdigest()


def reference_id(value: Any) -> str:
    """Resolve both legacy reference objects and canonical reference strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("ref"):
            return str(value["ref"])
        source = value.get("source_type")
        identifier = value.get("record_id") or value.get("source_id")
        if source and identifier:
            return f"{source}:{identifier}"
    return ""


def representative_nights(rows: list[dict]) -> list[dict]:
    """Keep the latest positive, negative and neutral/conflicting night."""
    selected = {}
    for row in sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda item: (str(item.get("night", "")), _json(item)),
    ):
        direction = row.get("direction", 0)
        direction = direction if type(direction) is int and direction in {-1, 0, 1} else 0
        selected[direction] = row
    return [
        deepcopy(row)
        for row in sorted(selected.values(), key=lambda item: str(item.get("night", "")))
    ]


def compact_profile_provenance(profile: dict[str, Any]) -> dict[str, Any]:
    """Project audit fields only; preserve all physiological and safety values.

    Original records remain authoritative. Counts/digests describe the input to
    this projection, not an immutable audit archive or a reconstructible snapshot.
    """
    result = deepcopy(profile)
    roots = {
        key: result[key]
        for key in ("based_on", "short_term_profile", "rolling_baseline", "profile_quality")
        if isinstance(result.get(key), dict)
    }
    based_on = roots.setdefault("based_on", {})
    result["based_on"] = based_on
    # An already compact view must be stable across repeated reads/replays.
    existing = based_on.get("provenance_summary")
    if isinstance(existing, dict) and existing.get("version") == PROVENANCE_VERSION:
        if (
            len(based_on.get("reference_table", [])) <= MAX_SOURCE_REFS
            and len(
                _json(
                    {
                        "based_on": based_on,
                        "nested_sources": _audit_fields(
                            {key: value for key, value in roots.items() if key != "based_on"}
                        ),
                    }
                )
            )
            <= MAX_PROVENANCE_BYTES
        ):
            return result
        raise ValueError("profile_provenance_budget_exceeded")
    references: dict[str, dict] = {}
    occurrences = 0
    audit_inputs = []
    important: set[str] = set()

    def visit(value: Any) -> Any:
        nonlocal occurrences
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        projected = {}
        for key, child in value.items():
            if key in _REF_KEYS and isinstance(child, list):
                child = [
                    entry
                    for entry in child
                    if reference_id(entry).split(":", 1)[0]
                    not in {"profile_short", "profile_long", "local_profile_short"}
                ]
                audit_inputs.append(child)
                ids = []
                for entry in child:
                    ref = reference_id(entry)
                    if not ref:
                        continue
                    occurrences += 1
                    ids.append(ref)
                    metadata = {"ref": ref}
                    if isinstance(entry, dict):
                        metadata.update(
                            {
                                field: entry[field]
                                for field in (
                                    "source_type",
                                    "record_id",
                                    "source_id",
                                    "revision",
                                    "content_hash",
                                    "record_date",
                                    "received_at_utc",
                                    "observed_at",
                                )
                                if field in entry
                            }
                        )
                    references.setdefault(ref, {"ref": ref}).update(metadata)
                projected["source_refs"] = sorted(set(ids))
            elif key == "night_votes" and isinstance(child, list):
                audit_inputs.append(child)
                projected["night_vote_count"] = len(child)
                representatives = representative_nights(child)
                for night in representatives:
                    important.update(reference_id(ref) for ref in night.get("source_refs", []))
                projected["night_votes"] = visit(representatives)
            elif key == "operation_bursts" and isinstance(child, list):
                audit_inputs.append(child)
                projected["operation_burst_count"] = len(child)
            elif key not in {"reference_table", "provenance_summary"}:
                projected[key] = visit(child)
        return projected

    for key, value in list(roots.items()):
        result[key] = visit(value)
    based_on = result["based_on"]

    # Prefer questionnaire and physiological summaries over unrelated legacy ops.
    def priority(ref: str) -> tuple[int, int, str]:
        return (
            0 if ref in important else 1,
            0
            if ref.split(":", 1)[0]
            in {
                "questionnaires",
                "questionnaire_answer",
                "wearable_daily",
                "sleep_reports",
                "records",
            }
            else 1,
            ref,
        )

    candidates = [
        ref for ref in references if ref in important or not ref.startswith("operations:")
    ]
    selected = sorted(candidates, key=priority)[:MAX_SOURCE_REFS]
    summary = {
        "version": PROVENANCE_VERSION,
        "input_reference_count": occurrences,
        "unique_reference_count": len(references),
        "retained_reference_count": len(selected),
        "digest": _digest(audit_inputs),
        "compacted": True,
        "references_omitted": len(references) > len(selected),
    }
    based_on["provenance_summary"] = summary

    def restrict(value: Any, allowed: set[str]) -> None:
        if isinstance(value, list):
            for item in value:
                restrict(item, allowed)
        elif isinstance(value, dict):
            if isinstance(value.get("source_refs"), list):
                value["source_refs"] = [ref for ref in value["source_refs"] if ref in allowed]
            for key, child in value.items():
                if key not in {"source_refs", "reference_table", "provenance_summary"}:
                    restrict(child, allowed)

    # References are stored once as metadata; use IDs everywhere else.
    while True:
        allowed = set(selected)
        for key in roots:
            restrict(result[key], allowed)
        based_on["reference_table"] = [references[ref] for ref in selected]
        summary["retained_reference_count"] = len(selected)
        # Bound all provenance, including repeated ID lists and legacy audit blobs.
        audit_view = {key: result[key] for key in ("based_on",) if key in result}
        audit_view["nested_sources"] = _audit_fields(
            {key: result[key] for key in roots if key != "based_on"}
        )
        if len(_json(audit_view)) <= MAX_PROVENANCE_BYTES - 512:
            break
        if not selected:
            # Do not silently truncate arbitrary business/unknown fields.
            raise ValueError("profile_provenance_budget_exceeded")
        selected = selected[: len(selected) // 2]
        summary["references_omitted"] = True
    summary["serialized_bytes"] = len(_json(audit_view))
    return result


def _audit_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [child for item in value if (child := _audit_fields(item))]
    if isinstance(value, dict):
        return {
            key: child
            for key, item in value.items()
            if (child := item if key in {"source_refs", "night_votes"} else _audit_fields(item))
        }
    return None
