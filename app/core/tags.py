"""六维标签前缀与扁平 string[] 互转（Mongo/comm 与 HTTP 六维对象边界）。"""

from __future__ import annotations

# 扁平 tag 前缀 → ES 六维字段名；无前缀默认归入 content_form
TAG_DIMENSION_PREFIXES: dict[str, str] = {
    "sleep:": "sleep_stage",
    "content:": "content_form",
    "mechanism:": "mechanism",
    "feat:": "audio_feat",
    "rhythm:": "rhythm",
    "risk:": "risk_control",
}

DIMENSION_TO_PREFIX: dict[str, str] = {
    dimension: prefix for prefix, dimension in TAG_DIMENSION_PREFIXES.items()
}

DIMENSION_FIELDS: tuple[str, ...] = (
    "sleep_stage",
    "content_form",
    "mechanism",
    "audio_feat",
    "rhythm",
    "risk_control",
)

DEFAULT_DIMENSION = "content_form"


def resolve_flat_tag(tag: str) -> tuple[str, str]:
    """扁平标签 → (维度字段名, label)。"""
    for prefix, dimension in TAG_DIMENSION_PREFIXES.items():
        if tag.startswith(prefix):
            return dimension, tag[len(prefix) :]
    return DEFAULT_DIMENSION, tag


def flat_tags_from_dimensions(dimensions: dict[str, list[str]]) -> list[str]:
    """六维 label 列表 → Mongo/comm 扁平 string[]。"""
    flat: list[str] = []
    for field in DIMENSION_FIELDS:
        prefix = DIMENSION_TO_PREFIX[field]
        for label in dimensions.get(field, []):
            if label:
                flat.append(f"{prefix}{label}")
    return flat


def dimensions_from_flat_tags(flat_tags: list[str]) -> dict[str, list[str]]:
    """Mongo/comm 扁平 string[] → 六维 label 列表。"""
    grouped: dict[str, list[str]] = {field: [] for field in DIMENSION_FIELDS}
    for tag in flat_tags:
        dimension, label = resolve_flat_tag(tag)
        if label:
            grouped[dimension].append(label)
    return grouped
