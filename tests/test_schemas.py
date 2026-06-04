"""Pydantic 模型与检索工具单元测试（不依赖 ES / gRPC / 模型）。"""

import pytest

from app.schemas.audio import AudioTags, EvidenceLevel, TagItem
from app.services.retrieval import _cosine_similarity


def test_audio_tags_content_labels() -> None:
    tags = AudioTags(
        content_form=[TagItem(vector_id="v1", label="白噪音")],
        mechanism=[TagItem(vector_id="v2", label="放松")],
    )
    labels = tags.content_labels()
    assert labels == {"白噪音", "放松"}


def test_cosine_similarity_identical() -> None:
    vec = [1.0, 0.0, 0.0]
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_evidence_level_values() -> None:
    assert EvidenceLevel.A.value == "A"
    assert EvidenceLevel.X.value == "X"
