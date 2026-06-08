"""tags 六维入参与扁平 string[] 互转单元测试。"""

import pytest

from app.core.tags import (
    dimensions_from_flat_tags,
    flat_tags_from_dimensions,
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


def _write_audio_json(**overrides: object) -> dict:
    body = {
        "category_code": 8,
        "noise_color": None,
        "level": 2,
        "name": "柔和鋼琴 篝火白噪音",
        "description": "",
        "tags": {
            "sleep_stage": ["放松"],
            "content_form": ["雨声"],
            "mechanism": [],
            "audio_feat": [],
            "rhythm": [],
            "risk_control": [],
        },
        "audio_info": {
            "meta_data": {
                "url": "https://cdn.fulai.tech/common/audio/20260527222706_k4tejx.mp3",
                "duration_sec": 600,
            },
            "is_loopable": True,
            "is_voice": False,
        },
        "evidence_level": "B",
        "recommend_weight": 0.75,
    }
    body.update(overrides)
    return body


def test_create_audio_request_accepts_tags_object() -> None:
    from app.schemas.audio import CreateAudioRequest

    req = CreateAudioRequest.model_validate(_write_audio_json())
    assert req.name == "柔和鋼琴 篝火白噪音"
    assert req.audio_url == "https://cdn.fulai.tech/common/audio/20260527222706_k4tejx.mp3"
    assert req.flat_tags() == ["sleep:放松", "content:雨声"]


def test_create_audio_request_rejects_legacy_flat_fields() -> None:
    from app.schemas.audio import CreateAudioRequest

    with pytest.raises(Exception):
        CreateAudioRequest.model_validate(
            {
                "audio_url": "https://cdn.example.com/a.mp3",
                "audio_name": "深夜雨声",
                "tags": ["sleep:放松", "content:雨声"],
            }
        )


def test_cosine_similarity_identical() -> None:
    vec = [1.0, 0.0, 0.0]
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_audio_material_data_from_comm_material() -> None:
    from app.bionode_grpc_clients.comm.grpc_gen import bionode_comm_pb2
    from app.schemas.audio import AudioMaterialData

    material = bionode_comm_pb2.AudioMaterialInfo(
        id="674a1b2c3d4e5f6789012345",
        category_code=0,
        level=2,
        noise_color="",
        name="深夜雨声",
        description="",
        tags=["sleep:放松", "content:雨声"],
        audio_info=bionode_comm_pb2.AudioMetaInfo(
            meta_data=bionode_comm_pb2.AudioMetaData(
                url="https://cdn.example.com/a.mp3",
                duration_sec=600,
            ),
            is_loopable=False,
            is_voice=False,
        ),
        status=1,
        create_time="2026-06-06T12:00:00Z",
        update_time="2026-06-06T12:00:00Z",
    )
    data = AudioMaterialData.from_comm_material(material)
    assert data.id == material.id
    assert data.name == "深夜雨声"
    assert data.level == 2
    assert data.tags == ["sleep:放松", "content:雨声"]
    assert data.audio_info.meta_data.url == "https://cdn.example.com/a.mp3"
    assert data.audio_info.meta_data.duration_sec == 600


def test_evidence_level_values() -> None:
    assert EvidenceLevel.A.value == "A"
    assert EvidenceLevel.X.value == "X"
