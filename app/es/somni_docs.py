"""Somni 音频原料文档 → ES 辅助（HTTP CUD 与定时同步共用）。"""

from __future__ import annotations

from typing import Any

DESCRIPTION_TAG_FIELDS = (
    "sleep_stage_tags",
    "content_form_tags",
    "audio_engineering_tags",
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


def _tag_labels(doc: dict[str, Any], *, keys: tuple[str, ...]) -> list[str]:
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
                for key in keys
                if item.get(key)
            )
    return [label for label in labels if label]


def build_material_description_text(doc: dict[str, Any]) -> str:
    """中文混合文本：audio_name + description + 标签 name/code/en_name。"""
    parts = [
        str(doc.get("audio_name", "")).strip(),
        str(doc.get("description", "")).strip(),
        " ".join(_tag_labels(doc, keys=("name", "code", "en_name"))).strip(),
    ]
    return " ".join(part for part in parts if part)


def build_material_description_text_en(doc: dict[str, Any]) -> str:
    """英文混合文本：audio_name_en + description_en + 标签 en_name。"""
    parts = [
        str(doc.get("audio_name_en", "")).strip(),
        str(doc.get("description_en", "")).strip(),
        " ".join(_tag_labels(doc, keys=("en_name",))).strip(),
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
    payload["description_text_en"] = build_material_description_text_en(payload)
    payload["sleep_stage_names"] = extract_sleep_stage_names(payload)
    return payload


def resolve_search_language(language: str | None) -> str:
    """请求/展示语言：缺省/zh→zh；en 及其它→en。"""
    lang = (language or "").strip().lower()
    if not lang or lang == "zh" or lang.startswith("zh"):
        return "zh"
    return "en"


def resolve_query_language(text: str, request_language: str | None = None) -> str:
    """按输入文字选检索字段/模型：有汉字→zh；否则有字母→en；否则回退请求语言。"""
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    if any("A" <= char <= "Z" or "a" <= char <= "z" for char in text):
        return "en"
    return resolve_search_language(request_language)


def description_text_field(language: str) -> str:
    """BM25 字段：zh→description_text，en→description_text_en。"""
    return (
        "description_text_en"
        if resolve_search_language(language) == "en"
        else "description_text"
    )


def description_vector_field(language: str, *, override: str = "") -> str:
    """knn 字段：zh→description_vector，en→description_vector_en。"""
    forced = (override or "").strip()
    if forced:
        return forced
    return (
        "description_vector_en"
        if resolve_search_language(language) == "en"
        else "description_vector"
    )
