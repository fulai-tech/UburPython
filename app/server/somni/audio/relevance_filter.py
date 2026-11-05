"""GetAudio 召回后相关过滤：分门槛、断崖截断、颜色噪音互斥。"""

from __future__ import annotations

from typing import Any

# 与手板 retrieval 一致：指定某一色后，兄弟色互斥（embedding 上极易混淆）
_COLOR_NOISE_CANONICAL = frozenset({"白噪音", "粉噪音", "棕噪音"})
_COLOR_NOISE_ALIASES: dict[str, str] = {
    "白噪音": "白噪音",
    "白噪": "白噪音",
    "白噪声": "白噪音",
    "白色噪音": "白噪音",
    "white noise": "白噪音",
    "whitenoise": "白噪音",
    "粉噪音": "粉噪音",
    "粉噪": "粉噪音",
    "粉噪声": "粉噪音",
    "粉红噪音": "粉噪音",
    "粉红噪声": "粉噪音",
    "粉色噪音": "粉噪音",
    "pink noise": "粉噪音",
    "pinknoise": "粉噪音",
    "棕噪音": "棕噪音",
    "棕噪": "棕噪音",
    "棕噪声": "棕噪音",
    "棕色噪音": "棕噪音",
    "褐噪音": "棕噪音",
    "褐色噪音": "棕噪音",
    "布朗噪音": "棕噪音",
    "brown noise": "棕噪音",
    "brownnoise": "棕噪音",
    "brownian noise": "棕噪音",
    "brownian": "棕噪音",
}
# 长别名优先，避免 white 误伤
_COLOR_ALIAS_KEYS = tuple(
    sorted(_COLOR_NOISE_ALIASES.keys(), key=len, reverse=True)
)


def detect_color_noise_intents(text: str) -> list[str]:
    """从 query 识别明确颜色噪音诉求，返回规范标签（可多值，极少见）。"""
    if not text:
        return []
    lowered = text.lower()
    found: list[str] = []
    for alias in _COLOR_ALIAS_KEYS:
        canonical = _COLOR_NOISE_ALIASES[alias]
        if alias in lowered and canonical not in found:
            found.append(canonical)
    return found


def _doc_title(doc: dict[str, Any]) -> str:
    return str(doc.get("audio_name") or "").lower()


def _doc_tag_blob(doc: dict[str, Any]) -> str:
    parts: list[str] = []
    tags = doc.get("content_form_tags") or doc.get("tag") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                parts.append(str(tag.get("name") or ""))
                parts.append(str(tag.get("name_en") or ""))
                parts.append(str(tag.get("en_name") or ""))
                parts.append(str(tag.get("code") or ""))
            else:
                parts.append(str(tag))
    return " ".join(parts).lower()


def _colors_mentioned_in_text(text: str) -> set[str]:
    found: set[str] = set()
    for alias in _COLOR_ALIAS_KEYS:
        if alias in text:
            found.add(_COLOR_NOISE_ALIASES[alias])
    # ES code：white_noise / pink_noise / brown_noise
    for code, canonical in (
        ("white_noise", "白噪音"),
        ("pink_noise", "粉噪音"),
        ("brown_noise", "棕噪音"),
    ):
        if code in text:
            found.add(canonical)
    return found


def filter_conflicting_color_noise(
    docs: list[dict[str, Any]],
    *,
    query_text: str,
) -> list[dict[str, Any]]:
    """
    查询明确点到某一色噪音时，剔除「标题/标签」声明了兄弟色、且未声明目标色的文档。

    只用标题 + content_form 标签判定色身份；短描述常串写其它色（或标签标错），
    不能用来把兄弟色结果保下来。
    """
    intents = set(detect_color_noise_intents(query_text)).intersection(
        _COLOR_NOISE_CANONICAL
    )
    if len(intents) != 1:
        return list(docs)
    preferred = next(iter(intents))
    siblings = _COLOR_NOISE_CANONICAL - {preferred}
    out: list[dict[str, Any]] = []
    for doc in docs:
        title_colors = _colors_mentioned_in_text(_doc_title(doc))
        tag_colors = _colors_mentioned_in_text(_doc_tag_blob(doc))
        # 标题已点名兄弟色 → 必删（即使描述/标签写成目标色）
        if title_colors & siblings and preferred not in title_colors:
            continue
        # 标题未点目标色，但标签点了兄弟色 → 删
        if (
            preferred not in title_colors
            and (tag_colors & siblings)
            and preferred not in tag_colors
        ):
            continue
        out.append(doc)
    return out


def diversify_multi_color_noise(
    docs: list[dict[str, Any]],
    *,
    query_text: str,
    per_color: int = 2,
) -> list[dict[str, Any]]:
    """多色意图时：每种色先各取 per_color 条，再按原序补齐其余。"""
    intents = [
        color
        for color in detect_color_noise_intents(query_text)
        if color in _COLOR_NOISE_CANONICAL
    ]
    if len(intents) < 2 or per_color <= 0:
        return list(docs)

    intent_set = set(intents)
    sibling_only = _COLOR_NOISE_CANONICAL - intent_set
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for color in intents:
        picked = 0
        for doc in docs:
            doc_id = _doc_id(doc)
            if not doc_id or doc_id in used:
                continue
            if color not in _doc_color_labels(doc):
                continue
            out.append(doc)
            used.add(doc_id)
            picked += 1
            if picked >= per_color:
                break
    for doc in docs:
        doc_id = _doc_id(doc)
        if not doc_id or doc_id in used:
            continue
        labels = _doc_color_labels(doc)
        # 未点名的兄弟色不补进结果（白+粉 时不要棕）
        if labels & sibling_only and not (labels & intent_set):
            continue
        out.append(doc)
        used.add(doc_id)
    return out


def _doc_id(doc: dict[str, Any]) -> str:
    return str(doc.get("_id") or doc.get("id") or "")


def _doc_color_labels(doc: dict[str, Any]) -> set[str]:
    blob = f"{_doc_title(doc)} {_doc_tag_blob(doc)}"
    return _colors_mentioned_in_text(blob)


def filter_docs_by_min_score(
    docs: list[dict[str, Any]],
    *,
    score_field: str,
    min_score: float,
) -> list[dict[str, Any]]:
    """绝对分门槛；无分数字段的文档保留（兼容单测/缺分场景）。"""
    if min_score <= 0:
        return list(docs)
    out: list[dict[str, Any]] = []
    for doc in docs:
        if score_field not in doc:
            out.append(doc)
            continue
        if float(doc[score_field]) >= min_score:
            out.append(doc)
    return out


def filter_docs_by_score_cliff(
    docs: list[dict[str, Any]],
    *,
    score_field: str,
    cliff_ratio: float,
) -> list[dict[str, Any]]:
    """
    相对断崖：保留 score >= top_score * cliff_ratio 的前缀段。
    docs 须已按分数降序；cliff_ratio∈(0,1]，1 表示只保留与最高分相同的。
    """
    if not docs or cliff_ratio <= 0:
        return list(docs)
    ratio = min(1.0, cliff_ratio)
    top: float | None = None
    out: list[dict[str, Any]] = []
    for doc in docs:
        if score_field not in doc:
            out.append(doc)
            continue
        score = float(doc[score_field])
        if top is None:
            top = score
            out.append(doc)
            continue
        if top <= 0:
            # 全 0 分：不断崖截断，避免误删无分/占位结果
            out.append(doc)
            continue
        if score >= top * ratio:
            out.append(doc)
        else:
            break
    return out


def apply_lane_relevance_filters(
    docs: list[dict[str, Any]],
    *,
    score_field: str,
    min_score: float,
    cliff_ratio: float,
    color_intents: list[str] | None = None,
) -> list[dict[str, Any]]:
    """先绝对门槛，再相对断崖；多色意图时按色分别断崖再合并。"""
    if not any(score_field in d for d in docs):
        return list(docs)
    ranked = sorted(
        docs,
        key=lambda d: float(d.get(score_field) or 0.0),
        reverse=True,
    )
    filtered = filter_docs_by_min_score(
        ranked, score_field=score_field, min_score=min_score
    )
    intents = [
        color
        for color in (color_intents or [])
        if color in _COLOR_NOISE_CANONICAL
    ]
    if len(intents) >= 2:
        return filter_docs_by_score_cliff_per_color(
            filtered,
            score_field=score_field,
            cliff_ratio=cliff_ratio,
            color_intents=intents,
        )
    return filter_docs_by_score_cliff(
        filtered, score_field=score_field, cliff_ratio=cliff_ratio
    )


def filter_docs_by_score_cliff_per_color(
    docs: list[dict[str, Any]],
    *,
    score_field: str,
    cliff_ratio: float,
    color_intents: list[str],
) -> list[dict[str, Any]]:
    """
    多色断崖：每种意图色在各自桶内相对截断后合并。
    不保留非意图色文档（雨声等），避免垫进多色噪音结果。
    """
    intent_set = {
        color for color in color_intents if color in _COLOR_NOISE_CANONICAL
    }
    if len(intent_set) < 2:
        return filter_docs_by_score_cliff(
            docs, score_field=score_field, cliff_ratio=cliff_ratio
        )

    kept: dict[str, dict[str, Any]] = {}
    for color in color_intents:
        if color not in intent_set:
            continue
        bucket = [doc for doc in docs if color in _doc_color_labels(doc)]
        ranked = sorted(
            bucket,
            key=lambda d: float(d.get(score_field) or 0.0),
            reverse=True,
        )
        for doc in filter_docs_by_score_cliff(
            ranked, score_field=score_field, cliff_ratio=cliff_ratio
        ):
            doc_id = _doc_id(doc)
            if doc_id:
                kept[doc_id] = doc

    return sorted(
        kept.values(),
        key=lambda d: float(d.get(score_field) or 0.0),
        reverse=True,
    )
