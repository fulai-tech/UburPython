"""EsSearch 新索引 parse_tags 映射与 mapping 注释单元测试。"""

from __future__ import annotations

from app.es.index_mappings import (
    build_handboard_audio_materials_mapping,
    build_handboard_audio_tag_dictionary_mapping,
    build_somni_audio_materials_mapping,
    build_somni_audio_tag_dictionary_mapping,
)
from app.es.search import EsSearch


def test_materials_mapping_fields_have_descriptions() -> None:
    mapping = build_somni_audio_materials_mapping(512, 384)
    props = mapping["properties"]
    assert props["audio_name"]["meta"]["description"]
    assert props["audio_name_en"]["type"] == "keyword"
    assert props["description_en"]["type"] == "text"
    assert props["description_text_en"]["type"] == "text"
    assert props["description_vector"]["dims"] == 512
    assert props["description_vector_en"]["dims"] == 384
    assert mapping["_meta"]["field_descriptions"]["sleep_stage_tags"]
    assert mapping["_meta"]["field_descriptions"]["sleep_stage_names"]
    assert props["sleep_stage_names"]["type"] == "keyword"
    assert props["content_form_tags"]["properties"]["en_name"]["meta"]["description"]
    assert props["audio_engineering_tags"]["properties"]["band_values"]["meta"]["description"]


def test_dictionary_mapping_vector_fields_have_descriptions() -> None:
    props = build_somni_audio_tag_dictionary_mapping(512, 384)["properties"]
    assert "512" in props["name_vector"]["meta"]["description"]
    assert props["name_en_vector"]["dims"] == 384
    assert "384" in props["name_en_vector"]["meta"]["description"]
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

    assert tags.sleep_stage[0].label == "unwind"
    assert tags.sleep_stage[0].vector_id == "s1"
    assert tags.content_form[0].label == "music"
    assert tags.mechanism == []
    assert tags.audio_feat[0].label == "event_density"
    assert tags.risk_control == []
    assert tags.rhythm == []


def test_content_tag_ids_collects_three_dimensions() -> None:
    raw = {
        "content_form_tags": [{"tag_id": "c1", "code": "x", "name": "音乐"}],
        "mechanism_tags": [{"tag_id": "m1", "code": "y", "name": "遮蔽"}],
        "audio_engineering_tags": [{"tag_id": "a1", "code": "z", "name": "密度"}],
    }
    tags = EsSearch.parse_tags(raw)
    assert EsSearch.content_tag_ids(tags) == ["c1", "a1"]


def test_handboard_mappings_keep_single_embedding_dim() -> None:
    materials = build_handboard_audio_materials_mapping(512)["properties"]
    tags = build_handboard_audio_tag_dictionary_mapping(512)["properties"]
    assert "description_vector_en" not in materials
    assert "audio_name_en" not in materials
    assert materials["description_vector"]["dims"] == 512
    assert tags["name_vector"]["dims"] == 512
    assert tags["name_en_vector"]["dims"] == 512


def test_somni_and_handboard_mappings_are_separate() -> None:
    handboard = build_handboard_audio_tag_dictionary_mapping(512)["properties"]
    somni = build_somni_audio_tag_dictionary_mapping(512, 384)["properties"]
    assert handboard["name_en_vector"]["dims"] == 512
    assert somni["name_en_vector"]["dims"] == 384
