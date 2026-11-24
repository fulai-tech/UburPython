"""双语混合文本与检索语言解析。"""

from app.es.somni_docs import (
    build_locale_description_text,
    build_material_description_text,
    build_material_description_text_en,
    build_tag_name_by_code_language,
    description_text_field,
    description_vector_field,
    locale_source_for_es,
    locale_vector_field,
    material_source_for_es,
    material_tag_labels_from_dictionary,
    resolve_locale_language,
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


def test_resolve_locale_language() -> None:
    assert resolve_locale_language("") == "zh"
    assert resolve_locale_language(None) == "zh"
    assert resolve_locale_language("zh-CN") == "zh"
    assert resolve_locale_language("en-US") == "en"
    assert resolve_locale_language("fr") == "fr"
    assert resolve_locale_language("ja") == "ja"
    assert locale_vector_field("zh") == "description_vector"
    assert locale_vector_field("en") == "description_vector_en"
    assert locale_vector_field("fr") is None
    assert locale_vector_field("ja") is None


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


def test_locale_description_text_appends_dictionary_tag_labels() -> None:
    name_by = build_tag_name_by_code_language(
        [
            {"code": "voice_guidance", "language": "zh", "name": "人声引导"},
            {"code": "voice_guidance", "language": "en", "name": "Voice Guidance"},
            {"code": "breathing_guidance", "language": "zh", "name": "呼吸引导"},
        ]
    )
    material = {
        "content_form_tags": [
            {"code": "voice_guidance"},
            {"code": "breathing_guidance"},
        ]
    }
    zh_labels = material_tag_labels_from_dictionary(
        material, language="zh", name_by_code_lang=name_by
    )
    en_labels = material_tag_labels_from_dictionary(
        material, language="en", name_by_code_lang=name_by
    )
    assert zh_labels == ["人声引导", "呼吸引导"]
    assert en_labels == ["Voice Guidance"]

    zh_text = build_locale_description_text(
        {
            "audio_name": "延长呼气呼吸引导",
            "description": "旁白引导呼吸",
            "summary": "",
        },
        tag_labels=zh_labels,
    )
    assert zh_text == "人声引导 呼吸引导"
    assert "延长呼气呼吸引导" not in zh_text
    assert "旁白引导呼吸" not in zh_text

    source = locale_source_for_es(
        {
            "material_id": "m1",
            "language": "zh",
            "audio_name": "延长呼气呼吸引导",
            "description": "旁白引导呼吸",
        },
        tag_labels=zh_labels,
    )
    assert source is not None
    assert source["description_text"] == "人声引导 呼吸引导"
