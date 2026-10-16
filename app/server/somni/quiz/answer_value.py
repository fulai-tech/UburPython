"""把答卷里的选项编码解析成用户选中的具体内容。"""

from __future__ import annotations

from typing import Any


def concrete_answer_value(input_type: str, value: Any, options: Any, extra_input: str = "") -> Any:
    if input_type in {"radio", "select"}:
        return _choice_scalar(value, _choice_items(options), extra_input)
    if input_type == "checkbox":
        seq = value if isinstance(value, list) else [value]
        return [_choice_scalar(item, _choice_items(options), extra_input) for item in seq]
    if input_type == "cascader":
        return _cascader_labels(value, _choice_items(options))
    if input_type == "composite":
        return _composite_values(value, options)
    if input_type == "switch" and isinstance(value, str):
        return _choice_scalar(value, _choice_items(options), extra_input)
    return value


def _choice_items(options: Any) -> list[dict]:
    nodes: list = []
    if isinstance(options, dict) and isinstance(options.get("items"), list):
        nodes = options["items"]
    elif isinstance(options, list):
        nodes = options
    items = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("format") or node.get("input_type"):
            continue
        if any(key in node for key in ("value", "label", "option_id", "option_text", "children")):
            items.append(node)
    return items


def _choice_scalar(token: Any, items: list[dict], extra_input: str) -> Any:
    code = token
    if isinstance(token, dict):
        free = str(token.get("input") or "").strip()
        if free:
            return free
        text = str(token.get("option_text") or token.get("label") or "").strip()
        if text:
            return text
        code = token.get("option_id", token.get("value", token.get("key")))
        if code is None:
            return token
    matched = _match_item(code, items)
    if matched is None:
        return code
    text = str(matched.get("label") or matched.get("option_text") or "").strip()
    if text:
        return text
    if (matched.get("is_input") or matched.get("is_input_enabled")) and extra_input.strip():
        return extra_input.strip()
    return code


def _match_item(code: Any, items: list[dict]) -> dict | None:
    wanted = str(code)
    for item in items:
        keys = [
            str(item[key])
            for key in ("value", "option_id", "key")
            if item.get(key) is not None and item.get(key) != ""
        ]
        if wanted in keys:
            return item
        children = item.get("children")
        if isinstance(children, list):
            found = _match_item(code, [child for child in children if isinstance(child, dict)])
            if found is not None:
                return found
    return None


def _cascader_labels(value: Any, items: list[dict]) -> Any:
    if isinstance(value, list):
        return [_cascader_node_label(node, items) for node in value]
    if isinstance(value, dict):
        labels = []
        node: Any = value
        while isinstance(node, dict):
            labels.append(_cascader_node_label(node, items))
            children = node.get("children")
            node = children[0] if isinstance(children, list) and children else None
        return labels
    return _choice_scalar(value, items, "")


def _cascader_node_label(node: Any, items: list[dict]) -> Any:
    if not isinstance(node, dict):
        return _choice_scalar(node, items, "")
    text = str(node.get("label") or node.get("option_text") or "").strip()
    if text:
        return text
    code = node.get("value", node.get("option_id"))
    if code is None:
        return node
    return _choice_scalar(code, items, str(node.get("input") or ""))


def _composite_values(value: Any, options: Any) -> Any:
    if not isinstance(value, list):
        return value
    subs = [item for item in options if isinstance(item, dict)] if isinstance(options, list) else []
    by_key = {str(sub["key"]): sub for sub in subs if sub.get("key") is not None}
    by_title = {str(sub["title"]): sub for sub in subs if sub.get("title")}
    resolved = []
    for index, child in enumerate(value):
        if not isinstance(child, dict):
            resolved.append(child)
            continue
        sub = None
        if child.get("key") is not None:
            sub = by_key.get(str(child["key"]))
        if sub is None and child.get("title"):
            sub = by_title.get(str(child["title"]))
        if sub is None and index < len(subs):
            sub = subs[index]
        input_type = str(child.get("input_type") or (sub or {}).get("input_type") or "")
        copied = dict(child)
        copied["value"] = concrete_answer_value(
            input_type,
            child.get("value"),
            (sub or {}).get("options"),
            str(child.get("extra_input") or ""),
        )
        resolved.append(copied)
    return resolved
