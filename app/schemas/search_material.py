from __future__ import annotations

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
        "mechanism_tags": _project_tag_refs(doc.get("mechanism_tags")),
        "audio_engineering_tags": _project_engineering_tags(doc.get("audio_engineering_tags")),
        "medical_risk_tags": _project_tag_refs(doc.get("medical_risk_tags")),
        "evidence_level_tags": _project_tag_refs(doc.get("evidence_level_tags")),
    }
    if "status" in doc:
        result["status"] = bool(doc["status"])
    return result


def _project_content_form_tags(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": item.get("name"),
                "parent_tag_id": item.get("parent_tag_id"),
                **({"code": reference_tag(item)} if reference_tag(item) else {}),
                **(
                    {"parent_tag_code": item["parent_tag_code"]}
                    if item.get("parent_tag_code")
                    else {}
                ),
            }
        )
    return out


def _project_engineering_tags(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        value_out = None
        if isinstance(value, dict):
            value_out = {"code": value.get("code")}
        out.append({"code": item.get("code"), "value": value_out})
    return out


def _project_tag_refs(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]
