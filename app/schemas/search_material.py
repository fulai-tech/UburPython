from __future__ import annotations

from typing import Any


def project_search_material(doc: dict[str, Any]) -> dict[str, Any]:
    material_id = str(doc.get("_id") or doc.get("id") or "")
    return {
        "_id": material_id,
        "audio_name": str(doc.get("audio_name") or ""),
        "description": str(doc.get("description") or ""),
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        "content_form_tags": _project_content_form_tags(doc.get("content_form_tags")),
        "audio_engineering_tags": _project_engineering_tags(
            doc.get("audio_engineering_tags")
        ),
    }


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
