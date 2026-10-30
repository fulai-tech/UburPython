from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.audio_tags import reference_tag


def project_search_material(doc: dict[str, Any]) -> dict[str, Any]:
    material_id = str(doc.get("_id") or doc.get("id") or "")
    result = {
        "_id": material_id,
        "audio_name": str(doc.get("audio_name") or ""),
        "description": str(doc.get("description") or ""),
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        "sleep_stage_tags": _project_tag_refs(doc.get("sleep_stage_tags")),
        "content_form_tags": _project_content_form_tags(doc.get("content_form_tags")),
        "audio_engineering_tags": _project_engineering_tags(doc.get("audio_engineering_tags")),
        "evidence_level_tags": _project_tag_refs(doc.get("evidence_level_tags")),
    }
    for key in (
        "short_description",
        "language",
        "languages",
        "source",
        "duration_seconds",
        "acoustic_metrics",
        "created_at",
        "updated_at",
    ):
        if key in doc:
            result[key] = deepcopy(doc[key])
    if "status" in doc:
        result["status"] = doc["status"] is True
    return result


def _project_content_form_tags(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        projected = deepcopy(item)
        if code := reference_tag(item):
            projected["code"] = code
        out.append(projected)
    return out


def _project_engineering_tags(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        projected = deepcopy(item)
        projected.setdefault("value", None)
        out.append(projected)
    return out


def _project_tag_refs(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]
