"""Pydantic 对外契约模型（替代 uburnode_audio.proto）。

字段 snake_case，与 ES / comm 全链路一致；检索出参列表字段用 audios。
"""

from enum import Enum
from typing import Self

from pydantic import BaseModel, Field

from app.core.tags import (
    DIMENSION_FIELDS,
    dimensions_from_flat_tags,
    flat_tags_from_dimensions,
)


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


class AudioTagsInput(BaseModel):
    """六维标签入参：各维为 label 字符串列表，字段名与 AudioTags 一致。"""

    sleep_stage: list[str] = Field(default_factory=list)
    content_form: list[str] = Field(default_factory=list)
    mechanism: list[str] = Field(default_factory=list)
    audio_feat: list[str] = Field(default_factory=list)
    rhythm: list[str] = Field(default_factory=list)
    risk_control: list[str] = Field(default_factory=list)

    def to_flat_tags(self) -> list[str]:
        """→ Mongo / comm 扁平 string[]。"""
        return flat_tags_from_dimensions(self.model_dump())

    @classmethod
    def from_flat_tags(cls, flat_tags: list[str]) -> Self:
        """Mongo / comm 扁平 string[] → 六维入参。"""
        return cls(**dimensions_from_flat_tags(flat_tags))

    def dimension_fields(self) -> tuple[str, ...]:
        return DIMENSION_FIELDS


class AudioTags(BaseModel):
    """六维标签结构，与 ES audio_materials.tags 同构（出参含 vector_id）。"""

    sleep_stage: list[TagItem] = Field(default_factory=list)
    content_form: list[TagItem] = Field(default_factory=list)
    mechanism: list[TagItem] = Field(default_factory=list)
    audio_feat: list[TagItem] = Field(default_factory=list)
    rhythm: list[TagItem] = Field(default_factory=list)
    risk_control: list[TagItem] = Field(default_factory=list)

    def content_labels(self) -> set[str]:
        """内容形态准入精确交集使用的标签集合（四维度并集）。"""
        labels: set[str] = set()
        for dim in (self.content_form, self.mechanism, self.audio_feat, self.rhythm):
            labels.update(item.label for item in dim)
        return labels

    def sleep_stage_labels(self) -> set[str]:
        return {item.label for item in self.sleep_stage}

    def to_label_tags(self) -> AudioTagsInput:
        """检索出参：六维标签仅保留 label 字符串。"""
        return AudioTagsInput(
            sleep_stage=[item.label for item in self.sleep_stage],
            content_form=[item.label for item in self.content_form],
            mechanism=[item.label for item in self.mechanism],
            audio_feat=[item.label for item in self.audio_feat],
            rhythm=[item.label for item in self.rhythm],
            risk_control=[item.label for item in self.risk_control],
        )


class AudioMetaDataIn(BaseModel):
    """创建/更新入参，与 comm AudioMetaData 同构。"""

    url: str
    duration_sec: int = 0


class AudioMetaInfoIn(BaseModel):
    """创建/更新入参，与 comm AudioMetaInfo 同构。"""

    meta_data: AudioMetaDataIn
    is_loopable: bool = False
    is_voice: bool = False


class WriteAudioRequest(BaseModel):
    """创建/更新音频共用请求体；tags 六维对象，服务端转扁平 string[] 写 comm / ES。"""

    category_code: int = 0
    noise_color: str | None = None
    level: int = 0
    name: str
    description: str = ""
    tags: AudioTagsInput = Field(default_factory=AudioTagsInput)
    audio_info: AudioMetaInfoIn
    evidence_level: EvidenceLevel = EvidenceLevel.C
    recommend_weight: float | None = None

    def flat_tags(self) -> list[str]:
        return self.tags.to_flat_tags()

    def resolved_noise_color(self) -> str:
        return self.noise_color or ""

    def resolved_recommend_weight(self) -> float:
        if self.recommend_weight is not None:
            return self.recommend_weight
        return EVIDENCE_WEIGHT_MAP[self.evidence_level]

    @property
    def audio_url(self) -> str:
        return self.audio_info.meta_data.url


class CreateAudioRequest(WriteAudioRequest):
    """POST /audio 请求体。"""


class UpdateAudioRequest(WriteAudioRequest):
    """PUT /audio/{material_id} 请求体，字段与创建一致；id 走路径参数。"""


class SearchAudioRequest(BaseModel):
    """POST /audio/search 请求体。"""

    sleep_stage_tags: list[str] = Field(default_factory=list)
    content_tags: list[str] = Field(default_factory=list)
    disliked_tags: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1)


class AudioResult(BaseModel):
    """单条检索结果。"""

    audio_url: str
    audio_name: str
    tags: AudioTagsInput
    evidence_level: EvidenceLevel
    recommend_weight: float


class AudioMetaDataOut(BaseModel):
    """与 comm AudioMetaData 同构。"""

    url: str = ""
    duration_sec: int = 0


class AudioMetaInfoOut(BaseModel):
    """与 comm AudioMetaInfo 同构。"""

    meta_data: AudioMetaDataOut = Field(default_factory=AudioMetaDataOut)
    is_loopable: bool = False
    is_voice: bool = False


class AudioMaterialData(BaseModel):
    """创建成功时 data，与 comm-service AudioMaterialInfo 同构。"""

    id: str
    category_code: int = 0
    level: int = 0
    noise_color: str = ""
    name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    audio_info: AudioMetaInfoOut = Field(default_factory=AudioMetaInfoOut)
    status: int = 0
    create_time: str = ""
    update_time: str = ""

    @classmethod
    def from_comm_material(cls, material: object) -> Self:
        """bionode_comm_pb2.AudioMaterialInfo → HTTP 出参。"""
        audio_info = getattr(material, "audio_info", None)
        meta_data = getattr(audio_info, "meta_data", None) if audio_info else None
        return cls(
            id=material.id,
            category_code=material.category_code,
            level=material.level,
            noise_color=material.noise_color,
            name=material.name,
            description=material.description,
            tags=list(material.tags),
            audio_info=AudioMetaInfoOut(
                meta_data=AudioMetaDataOut(
                    url=getattr(meta_data, "url", "") or "",
                    duration_sec=getattr(meta_data, "duration_sec", 0) or 0,
                ),
                is_loopable=getattr(audio_info, "is_loopable", False),
                is_voice=getattr(audio_info, "is_voice", False),
            ),
            status=material.status,
            create_time=material.create_time,
            update_time=material.update_time,
        )


class SearchAudioData(BaseModel):
    """检索成功时写入 ApiResponse.data。"""

    audios: list[AudioResult] = Field(default_factory=list)
