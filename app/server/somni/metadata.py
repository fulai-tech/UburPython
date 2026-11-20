"""量产音频语言参数解析。"""

from __future__ import annotations

_SUPPORTED_LANGUAGES = frozenset({"zh", "en", "fr", "ja"})
_DEFAULT_LANGUAGE = "zh"


def parse_language(raw: str | None) -> str:
    """解析 language 参数，支持 zh / en / fr / ja；空或缺省为 zh。"""
    value = (raw or "").strip().lower()
    if not value:
        return _DEFAULT_LANGUAGE
    if value not in _SUPPORTED_LANGUAGES:
        raise ValueError(f"language 仅支持 zh / en / fr / ja，当前为 {raw!r}")
    return value
