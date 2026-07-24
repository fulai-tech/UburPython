"""Somni 文档 → ES _source 辅助函数测试。"""

from __future__ import annotations

from app.es.somni_docs import extract_sleep_stage_names, material_source_for_es


def test_extract_sleep_stage_names_deduplicates_and_skips_blank() -> None:
    names = extract_sleep_stage_names(
        {
            "sleep_stage_tags": [
                {"tag_id": "s1", "code": "unwind", "name": "放松"},
                {"tag_id": "s2", "code": "soothe", "name": "入睡"},
                {"tag_id": "s1b", "code": "unwind", "name": "放松"},
                {"tag_id": "bad", "code": "x", "name": "  "},
                "invalid",
            ]
        }
    )
    assert names == ["放松", "入睡"]


def test_material_source_for_es_adds_sleep_stage_names() -> None:
    payload = material_source_for_es(
        {
            "id": "doc1",
            "audio_url": "https://cdn.example.com/a.mp3",
            "audio_name": "雨声",
            "description": "夜间雨声",
            "sleep_stage_tags": [{"tag_id": "s1", "code": "unwind", "name": "放松"}],
            "content_form_tags": [{"tag_id": "c1", "code": "rain", "name": "雨声"}],
        }
    )

    assert payload is not None
    assert "id" not in payload
    assert payload["sleep_stage_names"] == ["放松"]
    assert "雨声" in payload["description_text"]


def test_material_source_for_es_requires_audio_url() -> None:
    assert material_source_for_es({"audio_name": "无地址"}) is None
