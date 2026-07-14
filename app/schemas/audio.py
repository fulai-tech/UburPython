"""Pydantic 对外契约模型（替代 uburnode_audio.proto）。

字段 snake_case；检索出参直接返回 somni_audio_materials 索引文档（materials 列表）。
创建/更新入参对齐 Mongo Somni 文档结构。
"""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field

from app.core.tags import (
    DIMENSION_FIELDS,
    dimensions_from_flat_tags,
    flat_tags_from_dimensions,
)


class EvidenceLevel(StrEnum):
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
    """六维标签结构，与 ES 音频素材索引 tags 同构（出参含 vector_id）。"""

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
    """comm AudioMetaData 同构（删除等遗留路径仍可能使用）。"""

    url: str
    duration_sec: int = 0


class AudioMetaInfoIn(BaseModel):
    """comm AudioMetaInfo 同构（删除等遗留路径仍可能使用）。"""

    meta_data: AudioMetaDataIn
    is_loopable: bool = False
    is_voice: bool = False


class SomniTagRef(BaseModel):
    """嵌套标签通用字段（tag_id / code / name）。"""

    tag_id: str | None = None
    code: str | None = None
    name: str | None = None


class ContentFormTag(SomniTagRef):
    """内容形态标签（可含英文名与父级）。"""

    en_name: str | None = None
    parent_tag_id: str | None = None
    parent_tag_code: str | None = None


class AudioEngineeringTag(SomniTagRef):
    """音频工程特征（可含取值与频谱附加字段）。"""

    value: SomniTagRef | None = None
    band_values: list[float] | None = None
    relative_loudness: float | None = None


class SomniAudioBody(BaseModel):
    """somni_audio_materials 文档写字段（不含 id）；创建/更新共用形状。"""

    audio_url: str | None = None
    description: str | None = None
    operation_type: int | None = None
    created_by: str | None = None
    updated_by: str | None = None
    status: bool | None = None
    sleep_stage_tags: list[SomniTagRef] = Field(default_factory=list)
    content_form_tags: list[ContentFormTag] = Field(default_factory=list)
    mechanism_tags: list[SomniTagRef] = Field(default_factory=list)
    audio_engineering_tags: list[AudioEngineeringTag] = Field(default_factory=list)
    medical_risk_tags: list[SomniTagRef] = Field(default_factory=list)
    evidence_level_tags: list[SomniTagRef] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class CreateAudioRequest(SomniAudioBody):
    """POST /audio：仅 audio_name 必填，其余选填。"""

    audio_name: str = Field(min_length=1)

    def to_mongo_doc(self) -> dict[str, Any]:
        """转 Mongo 插入文档（去掉值为 None 的标量字段）。"""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class UpdateAudioRequest(SomniAudioBody):
    """PUT /audio/{material_id}：全部字段可选（含 audio_name）。"""

    audio_name: str | None = None

    def to_update_fields(self) -> dict[str, Any]:
        """仅包含请求中出现的字段，供 Mongo `$set`。"""
        return self.model_dump(exclude_unset=True)


class SearchAudioRequest(BaseModel):
    """POST /audio/search 请求体。"""

    query_text: str | None = None
    sleep_stage_tags: list[str] = Field(default_factory=list)
    content_tags: list[str] = Field(default_factory=list)
    disliked_tags: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1)


class SearchAudioData(BaseModel):
    """检索成功时写入 ApiResponse.data；每项为 somni_audio_materials 索引文档（含 id）。"""

    materials: list[dict[str, Any]] = Field(default_factory=list)


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
    """创建成功时 data，与 comm-service AudioMaterialInfo 同构（遗留）。"""

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
