"""EsSearch 新索引 parse_tags 映射与 mapping 注释单元测试。"""

from __future__ import annotations

from app.es.index_mappings import (
    build_somni_audio_materials_mapping,
    build_somni_audio_tag_dictionary_mapping,
)
from app.es.search import EsSearch


def test_materials_mapping_fields_have_descriptions() -> None:
    mapping = build_somni_audio_materials_mapping(512)
    props = mapping["properties"]
    assert props["audio_name"]["meta"]["description"]
    assert mapping["_meta"]["field_descriptions"]["sleep_stage_tags"]
    assert props["content_form_tags"]["properties"]["en_name"]["meta"]["description"]
    assert props["audio_engineering_tags"]["properties"]["band_values"]["meta"]["description"]


def test_dictionary_mapping_vector_fields_have_descriptions() -> None:
    props = build_somni_audio_tag_dictionary_mapping(512)["properties"]
    assert "512" in props["name_vector"]["meta"]["description"]
    assert props["name_en_vector"]["dims"] == 512
    assert props["created_by"]["meta"]["description"] == "标签创建人"
    assert props["updated_by"]["meta"]["description"] == "标签最后更新人"


def test_parse_tags_maps_somni_material_structure() -> None:
    raw = {
        "sleep_stage_tags": [{"tag_id": "s1", "code": "unwind", "name": "放松"}],
        "content_form_tags": [{"tag_id": "c1", "code": "music", "name": "音乐"}],
        "mechanism_tags": [{"tag_id": "m1", "code": "masking", "name": "遮蔽"}],
        "audio_engineering_tags": [
            {"tag_id": "a1", "code": "event_density", "name": "密度", "value": {"code": "low"}}
        ],
        "medical_risk_tags": [{"tag_id": "r1", "code": "caution", "name": "谨慎"}],
    }
    tags = EsSearch.parse_tags(raw)

    assert tags.sleep_stage[0].label == "放松"
    assert tags.sleep_stage[0].vector_id == "s1"
    assert tags.content_form[0].label == "音乐"
    assert tags.mechanism[0].label == "遮蔽"
    assert tags.audio_feat[0].label == "密度"
    assert tags.risk_control[0].label == "谨慎"
    assert tags.rhythm == []


def test_content_tag_ids_collects_three_dimensions() -> None:
    raw = {
        "content_form_tags": [{"tag_id": "c1", "code": "x", "name": "音乐"}],
        "mechanism_tags": [{"tag_id": "m1", "code": "y", "name": "遮蔽"}],
        "audio_engineering_tags": [{"tag_id": "a1", "code": "z", "name": "密度"}],
    }
    tags = EsSearch.parse_tags(raw)
    assert EsSearch.content_tag_ids(tags) == ["c1", "m1", "a1"]
