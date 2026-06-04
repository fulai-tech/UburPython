"""Pydantic 对外契约模型（替代 uburnode_audio.proto）。

字段 snake_case，与 ES / comm 全链路一致；列表字段用复数资源名（results 非 list）。
"""

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceLevel(str, Enum):
    """证据等级 A/B/C/D/R/X，与 ES keyword 及默认 recommend_weight 映射。"""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    R = "R"
    X = "X"


# 精排权重默认值；业务字段完善前精排序等效于 match_count（规范 §五-4）
EVIDENCE_WEIGHT_MAP: dict[EvidenceLevel, float] = {
    EvidenceLevel.A: 1.0,
    EvidenceLevel.B: 0.75,
    EvidenceLevel.C: 0.45,
    EvidenceLevel.D: 0.2,
    EvidenceLevel.R: 0.1,
    EvidenceLevel.X: 0.0,
}


class TagItem(BaseModel):
    """ES 六维标签子项：vector_id 关联 tag_vectors 索引。"""

    vector_id: str
    label: str


class AudioTags(BaseModel):
    """六维标签结构，与 ES audio_materials.tags 同构。"""

    sleep_stage: list[TagItem] = Field(default_factory=list)
    content_form: list[TagItem] = Field(default_factory=list)
    mechanism: list[TagItem] = Field(default_factory=list)
    audio_feat: list[TagItem] = Field(default_factory=list)
    rhythm: list[TagItem] = Field(default_factory=list)
    risk_control: list[TagItem] = Field(default_factory=list)

    def content_labels(self) -> set[str]:
        """内容形态准入 / 厌恶剔除使用的标签集合（四维度并集）。"""
        labels: set[str] = set()
        for dim in (self.content_form, self.mechanism, self.audio_feat, self.rhythm):
            labels.update(item.label for item in dim)
        return labels

    def sleep_stage_labels(self) -> set[str]:
        return {item.label for item in self.sleep_stage}


class CreateAudioRequest(BaseModel):
    """POST /audio 请求体。tags 为 Mongo 侧扁平 string[]，EsSync 负责拆六维。"""

    audio_url: str
    audio_name: str
    tags: list[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.C
    recommend_weight: float | None = None
    category_code: int = 0
    noise_color: str = ""
    description: str = ""


class UpdateAudioRequest(BaseModel):
    """PUT /audio/{id} 请求体；未传字段表示不更新。"""

    audio_url: str | None = None
    audio_name: str | None = None
    tags: list[str] | None = None
    evidence_level: EvidenceLevel | None = None
    recommend_weight: float | None = None
    category_code: int | None = None
    noise_color: str | None = None
    description: str | None = None
    status: int | None = None


class SearchAudioRequest(BaseModel):
    """POST /audio/search 请求体。"""

    sleep_stage_tags: list[str] = Field(default_factory=list)
    content_tags: list[str] = Field(default_factory=list)
    disliked_tags: list[str] = Field(default_factory=list)
    top_k: int = 10


class AudioResult(BaseModel):
    """单条检索结果。"""

    audio_url: str
    audio_name: str
    tags: AudioTags
    evidence_level: EvidenceLevel
    recommend_weight: float


class CreateAudioResponse(BaseModel):
    id: str


class SearchAudioResponse(BaseModel):
    results: list[AudioResult] = Field(default_factory=list)


class EmptyResponse(BaseModel):
    ok: bool = True
