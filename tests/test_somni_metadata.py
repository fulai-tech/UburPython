"""量产音频 language 参数解析。"""

from __future__ import annotations

import pytest

from app.server.somni.metadata import parse_language


def test_parse_language_defaults_to_zh() -> None:
    assert parse_language(None) == "zh"
    assert parse_language("") == "zh"
    assert parse_language("  ") == "zh"


def test_parse_language_accepts_zh_and_en() -> None:
    assert parse_language("zh") == "zh"
    assert parse_language("EN") == "en"


def test_parse_language_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="language"):
        parse_language("fr")
