"""tags 六维入参与扁平 string[] 互转单元测试。"""

import pytest

from app.core.tags import (
    dimensions_from_flat_tags,
    resolve_flat_tag,
)
from app.schemas.audio import AudioTags, AudioTagsInput, EvidenceLevel, TagItem
from app.services.retrieval import _cosine_similarity


def test_audio_tags_content_labels() -> None:
    tags = AudioTags(
        content_form=[TagItem(vector_id="v1", label="白噪音")],
        mechanism=[TagItem(vector_id="v2", label="放松")],
    )
    labels = tags.content_labels()
    assert labels == {"白噪音", "放松"}


def test_resolve_flat_tag_with_prefix() -> None:
    assert resolve_flat_tag("sleep:放松") == ("sleep_stage", "放松")
    assert resolve_flat_tag("content:雨声") == ("content_form", "雨声")
    assert resolve_flat_tag("feat:432Hz") == ("audio_feat", "432Hz")


def test_resolve_flat_tag_without_prefix() -> None:
    assert resolve_flat_tag("雨声") == ("content_form", "雨声")


def test_audio_tags_input_to_flat_tags() -> None:
    tags = AudioTagsInput(
        sleep_stage=["放松"],
        content_form=["雨声", "森林"],
        audio_feat=["低频持续"],
    )
    flat = tags.to_flat_tags()
    assert flat == ["sleep:放松", "content:雨声", "content:森林", "feat:低频持续"]


def test_audio_tags_input_from_flat_tags_roundtrip() -> None:
    flat = ["sleep:入睡", "content:下雨的声音", "mechanism:正念", "rhythm:缓慢"]
    restored = AudioTagsInput.from_flat_tags(flat)
    assert restored.to_flat_tags() == flat


def test_flat_tags_from_dimensions_empty_labels_skipped() -> None:
    grouped = dimensions_from_flat_tags(["sleep:守护", "content:白噪音"])
    assert grouped["sleep_stage"] == ["守护"]
    assert grouped["content_form"] == ["白噪音"]
    assert grouped["mechanism"] == []


def _somni_create_json(**overrides: object) -> dict:
    body: dict = {
        "audio_name": "阴雨天城市公寓的雷雨氛围感音效",
        "audio_url": "https://cdn.fulai.tech/somni/audio/demo.mp3",
        "operation_type": 0,
        "created_by": "qwen3.5-omni-plus",
        "updated_by": "qwen3.5-omni-plus",
        "description": "雨夜场景",
        "sleep_stage_tags": [
            {"tag_id": "s1", "code": "unwind", "name": "放松"},
        ],
        "content_form_tags": [
            {
                "tag_id": "c1",
                "code": "natural_sound",
                "name": "自然声",
                "en_name": "Natural Sound",
            }
        ],
        "mechanism_tags": [],
        "audio_engineering_tags": [
            {
                "tag_id": "e1",
                "code": "event_density",
                "name": "声音事件密度",
                "value": {"tag_id": "v1", "code": "medium_low", "name": "中低"},
            }
        ],
        "medical_risk_tags": [],
        "evidence_level_tags": [{"tag_id": "ev1", "code": "B", "name": "中等证据"}],
        "embedding": [0.1, 0.2],
    }
    body.update(overrides)
    return body


def test_create_audio_request_requires_only_audio_name() -> None:
    from app.schemas.audio import CreateAudioRequest

    req = CreateAudioRequest.model_validate({"audio_name": "仅名称即可"})
    assert req.audio_name == "仅名称即可"
    assert req.audio_url is None
    assert req.sleep_stage_tags == []


def test_create_audio_request_rejects_missing_audio_name() -> None:
    from pydantic import ValidationError

    from app.schemas.audio import CreateAudioRequest

    with pytest.raises(ValidationError):
        CreateAudioRequest.model_validate({"audio_url": "https://cdn.example.com/a.mp3"})


def test_create_audio_request_accepts_somni_document() -> None:
    from app.schemas.audio import CreateAudioRequest

    req = CreateAudioRequest.model_validate(_somni_create_json())
    assert req.audio_name.startswith("阴雨天")
    assert req.sleep_stage_tags[0].name == "放松"
    assert req.audio_engineering_tags[0].value is not None
    assert req.audio_engineering_tags[0].value.code == "medium_low"
    assert req.embedding == [0.1, 0.2]


def test_update_audio_request_all_fields_optional() -> None:
    from app.schemas.audio import UpdateAudioRequest

    req = UpdateAudioRequest.model_validate({})
    assert req.model_dump(exclude_unset=True) == {}

    partial = UpdateAudioRequest.model_validate({"description": "改描述"})
    assert partial.model_dump(exclude_unset=True) == {"description": "改描述"}


def test_cosine_similarity_identical() -> None:
    vec = [1.0, 0.0, 0.0]
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_audio_material_data_from_material_like() -> None:
    from types import SimpleNamespace

    from app.schemas.audio import AudioMaterialData

    material = SimpleNamespace(
        id="674a1b2c3d4e5f6789012345",
        description="夜雨",
        status=True,
        create_time="2026-06-06T12:00:00Z",
        update_time="2026-06-06T12:00:00Z",
        audio_name="深夜雨声",
        audio_url="https://cdn.example.com/a.mp3",
        operation_type=0,
        created_by="agent",
        updated_by="",
        sleep_stage_tags=[
            SimpleNamespace(tag_id="t1", code="unwind", name="放松")
        ],
        content_form_tags=[],
        mechanism_tags=[],
        audio_engineering_tags=[],
        medical_risk_tags=[],
        evidence_level_tags=[],
    )
    data = AudioMaterialData.from_material_like(material)
    assert data.id == material.id
    assert data.audio_name == "深夜雨声"
    assert data.audio_url == "https://cdn.example.com/a.mp3"
    assert data.status is True
    assert data.sleep_stage_tags == [
        {"tag_id": "t1", "code": "unwind", "name": "放松"}
    ]


def test_evidence_level_values() -> None:
    assert EvidenceLevel.A.value == "A"
    assert EvidenceLevel.X.value == "X"
