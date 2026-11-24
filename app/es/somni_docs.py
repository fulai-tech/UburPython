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
    """中文混合文本（语种纯净）：audio_name + description + 标签中文名。"""
    parts = [
        str(doc.get("audio_name", "")).strip(),
        str(doc.get("description", "")).strip(),
        " ".join(_tag_labels(doc, keys=("name",))).strip(),
    ]
    return " ".join(part for part in parts if part)


def build_material_description_text_en(doc: dict[str, Any]) -> str:
    """英文混合文本（语种纯净）：audio_name_en + description_en + 标签英文名。"""
    parts = [
        str(doc.get("audio_name_en", "")).strip(),
        str(doc.get("description_en", "")).strip(),
        " ".join(_tag_labels(doc, keys=("en_name",))).strip(),
    ]
    return " ".join(part for part in parts if part)


def material_source_for_es(doc: dict[str, Any]) -> dict[str, Any] | None:
    """手板 Mongo/HTTP 文档 → ES _source（无 _id/id）；缺 audio_url 返回 None。"""
    audio_url = str(doc.get("audio_url", "")).strip()
    if not audio_url:
        return None
    payload = {k: v for k, v in doc.items() if k not in ("_id", "id")}
    payload["audio_url"] = audio_url
    payload["description_text"] = build_material_description_text(payload)
    payload["description_text_en"] = build_material_description_text_en(payload)
    payload["sleep_stage_names"] = extract_sleep_stage_names(payload)
    return payload


_SOMNI_MATERIAL_DROPPED_FIELDS = (
    "_id",
    "id",
    "embedding",
    "language",
    "audio_name",
    "audio_name_en",
    "description",
    "description_en",
    "summary",
    "short_description",
    "short_description_en",
    "audio_url",
    "cover_url",
    "description_text",
    "description_text_en",
    "description_vector",
    "description_vector_en",
    "mechanism_tags",
    "medical_risk_tags",
)


def has_indexable_locale(rows: list[dict[str, Any]]) -> bool:
    """语言表至少有一行 language 非空，原料才进 ES 原料索引。"""
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("language") or "").strip():
            return True
    return False


def somni_material_source_for_es(doc: dict[str, Any]) -> dict[str, Any] | None:
    """量产原料 → ES _source。不含语言表字段；标题/地址按 material_id+language 查语言索引。"""
    doc_id = str(doc.get("_id") or doc.get("id") or "").strip()
    if not doc_id:
        return None
    payload = {k: v for k, v in doc.items() if k not in _SOMNI_MATERIAL_DROPPED_FIELDS}
    payload["sleep_stage_names"] = extract_sleep_stage_names(payload)
    return payload


def build_tag_name_by_code_language(
    tag_docs: list[dict[str, Any]],
    *,
    tag_type: str = "content_form",
) -> dict[tuple[str, str], str]:
    """标签词典 → (code, language) → 展示名。

    默认只收 content_form，避免同 code 跨 type 互相覆盖。
    同键后者覆盖。
    """
    wanted = str(tag_type or "").strip()
    out: dict[tuple[str, str], str] = {}
    for row in tag_docs:
        if not isinstance(row, dict):
            continue
        if wanted and str(row.get("type") or "").strip() != wanted:
            continue
        code = str(row.get("code") or "").strip()
        name = str(row.get("name") or "").strip()
        if not code or not name:
            continue
        lang = resolve_locale_language(str(row.get("language") or ""))
        out[(code, lang)] = name
    return out


def material_content_form_tag_codes(material: dict[str, Any] | None) -> list[str]:
    """原料 content_form_tags 上的 code，去重且保序。"""
    if not material:
        return []
    codes: list[str] = []
    seen: set[str] = set()
    items = material.get("content_form_tags") or []
    if not isinstance(items, list):
        return []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def material_tag_labels_from_dictionary(
    material: dict[str, Any] | None,
    *,
    language: str,
    name_by_code_lang: dict[tuple[str, str], str],
) -> list[str]:
    """按语言表 language 从词典取内容形态标签展示名；缺该语言名时跳过该 code。"""
    lang = resolve_locale_language(language)
    labels: list[str] = []
    seen: set[str] = set()
    for code in material_content_form_tag_codes(material):
        name = str(name_by_code_lang.get((code, lang)) or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        labels.append(name)
    return labels


def build_locale_description_text(
    doc: dict[str, Any],
    *,
    tag_labels: list[str] | tuple[str, ...] = (),
) -> str:
    """语言行检索文本：仅内容形态标签展示名（不含标题/描述/短描述）。"""
    del doc
    return " ".join(str(label).strip() for label in tag_labels if str(label).strip()).strip()


def locale_source_for_es(
    doc: dict[str, Any],
    *,
    tag_labels: list[str] | tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """语言表文档 → ES _source。缺 material_id 或 language 则跳过。"""
    material_id = str(doc.get("material_id", "")).strip()
    language = str(doc.get("language", "")).strip().lower()
    if not material_id or not language:
        return None
    summary = str(doc.get("summary") or doc.get("short_description") or "")
    payload = {
        "material_id": material_id,
        "language": language,
        "audio_name": str(doc.get("audio_name") or ""),
        "description": str(doc.get("description") or ""),
        "summary": summary,
        "audio_url": str(doc.get("audio_url") or ""),
        "cover_url": str(doc.get("cover_url") or ""),
        "create_time": doc.get("create_time"),
        "update_time": doc.get("update_time"),
    }
    payload["description_text"] = build_locale_description_text(
        payload, tag_labels=tag_labels
    )
    return payload


def resolve_search_language(language: str | None) -> str:
    """编码器 / 手板双字段：缺省/zh→zh；其它→en（仅中英两套模型）。"""
    lang = (language or "").strip().lower()
    if not lang or lang == "zh" or lang.startswith("zh"):
        return "zh"
    return "en"


def resolve_locale_language(language: str | None) -> str:
    """语言表 term 过滤：保留 zh/en/fr/ja 等主语言码；缺省→zh。"""
    lang = (language or "").strip().lower()
    if not lang or lang == "zh" or lang.startswith("zh"):
        return "zh"
    if lang == "en" or lang.startswith("en"):
        return "en"
    primary = lang.split("-", 1)[0].strip()
    return primary or "zh"


def resolve_query_language(text: str, request_language: str | None = None) -> str:
    """按输入文字选检索字段/模型：有汉字→zh；否则有字母→en；否则回退请求语言。"""
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    if any("A" <= char <= "Z" or "a" <= char <= "z" for char in text):
        return "en"
    return resolve_search_language(request_language)


def locale_vector_field(language: str) -> str | None:
    """语言表 knn 字段：仅 zh/en 有向量；其它语言返回 None（走 BM25）。"""
    lang = resolve_locale_language(language)
    if lang == "zh":
        return "description_vector"
    if lang == "en":
        return "description_vector_en"
    return None


def description_text_field(language: str) -> str:
    """手板 BM25 字段：zh→description_text，en→description_text_en。量产语言表固定用 description_text。"""
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
