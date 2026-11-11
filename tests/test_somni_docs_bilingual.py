"""双语混合文本与检索语言解析。"""

from app.es.somni_docs import (
    build_material_description_text,
    build_material_description_text_en,
    description_text_field,
    description_vector_field,
    material_source_for_es,
    resolve_query_language,
    resolve_search_language,
)


def test_resolve_search_language() -> None:
    assert resolve_search_language("") == "zh"
    assert resolve_search_language(None) == "zh"
    assert resolve_search_language("zh") == "zh"
    assert resolve_search_language("zh-CN") == "zh"
    assert resolve_search_language("en") == "en"
    assert resolve_search_language("fr") == "en"


def test_resolve_query_language() -> None:
    assert resolve_query_language("下雨声", "en") == "zh"
    assert resolve_query_language("ASMR 雨声", "en") == "zh"
    assert resolve_query_language("Theta波", "en") == "zh"
    assert resolve_query_language("rain", "zh") == "en"
    assert resolve_query_language("white noise", "zh") == "en"
    assert resolve_query_language("478", "en") == "en"
    assert resolve_query_language("478", "zh") == "zh"
    assert resolve_query_language("", "en") == "en"


def test_build_material_description_text_zh_and_en() -> None:
    doc = {
        "audio_name": "听涧",
        "audio_name_en": "Listening to the Stream",
        "description": "雨声助眠",
        "description_en": "Rain for sleep",
        "audio_url": "https://cdn.example.com/a.mp3",
        "content_form_tags": [
            {"name": "雨声", "code": "rain", "en_name": "Rain"},
        ],
    }
    zh = build_material_description_text(doc)
    assert "听涧" in zh
    assert "雨声助眠" in zh
    assert "雨声" in zh
    assert "rain" not in zh.lower()
    assert "Rain" not in zh

    en = build_material_description_text_en(doc)
    assert "Listening to the Stream" in en
    assert "Rain for sleep" in en
    assert "Rain" in en
    assert "听涧" not in en
    assert "雨声" not in en


def test_material_source_for_es_writes_both_texts() -> None:
    payload = material_source_for_es(
        {
            "audio_name": "听涧",
            "audio_name_en": "Stream",
            "description": "雨",
            "description_en": "rain",
            "audio_url": "https://cdn.example.com/a.mp3",
            "content_form_tags": [{"name": "雨声", "en_name": "Rain"}],
        }
    )
    assert payload is not None
    assert payload["description_text"]
    assert payload["description_text_en"]
    assert "Stream" in payload["description_text_en"]


def test_description_fields_by_language() -> None:
    assert description_text_field("zh") == "description_text"
    assert description_text_field("en") == "description_text_en"
    assert description_vector_field("zh") == "description_vector"
    assert description_vector_field("en") == "description_vector_en"
    assert description_vector_field("en", override="description_vector") == (
        "description_vector"
    )
