"""Somni 音频原料文档 → ES 辅助（HTTP CUD 与定时同步共用）。"""

from __future__ import annotations

from typing import Any

DESCRIPTION_TAG_FIELDS = (
    "sleep_stage_tags",
    "content_form_tags",
    "mechanism_tags",
    "audio_engineering_tags",
    "medical_risk_tags",
    "evidence_level_tags",
)


def extract_sleep_stage_names(doc: dict[str, Any]) -> list[str]:
    """从 sleep_stage_tags 提取中文名列表，供 ES 扁平 term 过滤。"""
    names: list[str] = []
    for item in doc.get("sleep_stage_tags") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def build_material_description_text(doc: dict[str, Any]) -> str:
    """拼接 audio_name + description + 标签 name/code/en_name。"""
    labels: list[str] = []
    for field in DESCRIPTION_TAG_FIELDS:
        items = doc.get(field) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            labels.extend(
                str(item.get(key, "")).strip()
                for key in ("name", "code", "en_name")
                if item.get(key)
            )
    parts = [
        str(doc.get("audio_name", "")).strip(),
        str(doc.get("description", "")).strip(),
        " ".join(label for label in labels if label).strip(),
    ]
    return " ".join(part for part in parts if part)


def material_source_for_es(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Mongo/HTTP 文档 → ES _source（无 _id/id）；缺 audio_url 返回 None。"""
    audio_url = str(doc.get("audio_url", "")).strip()
    if not audio_url:
        return None
    payload = {k: v for k, v in doc.items() if k not in ("_id", "id")}
    payload["audio_url"] = audio_url
    payload["description_text"] = build_material_description_text(payload)
    payload["sleep_stage_names"] = extract_sleep_stage_names(payload)
    return payload
