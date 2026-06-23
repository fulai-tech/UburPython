"""Somni ES 索引 mapping 定义（含字段含义注释，来源：音频表结构.md）。

注释写入 Elasticsearch mapping 的 meta.description，新建索引时生效。
"""

from __future__ import annotations

from typing import Any


_META_DESC_MAX_LEN = 50


def _short_meta(description: str) -> str:
    if len(description) <= _META_DESC_MAX_LEN:
        return description
    return description[: _META_DESC_MAX_LEN - 1] + "…"


def _field(field_type: str, description: str, **extra: Any) -> dict[str, Any]:
    return {"type": field_type, "meta": {"description": _short_meta(description)}, **extra}


def _nested(properties: dict[str, Any]) -> dict[str, Any]:
    """nested 类型（ES 不允许在 nested 映射上设置 meta）。"""
    return {"type": "nested", "properties": properties}


def _tag_ref_fields() -> dict[str, Any]:
    """嵌套标签项通用字段（tag_id / code / name）。"""
    return {
        "tag_id": _field("keyword", "标签 ID，关联 somni_audio_tag_dictionary 文档 _id"),
        "code": _field("keyword", "标签编码，用于规则判断、推荐策略与接口传输"),
        "name": _field("keyword", "标签中文名称，用于展示与检索精确匹配"),
    }


def build_somni_audio_materials_mapping(embedding_dim: int) -> dict[str, Any]:
    """somni_audio_materials 索引 mapping（与 Mongo 原料表 1:1）。"""
    tag_ref = _tag_ref_fields()
    return {
        "_meta": {
            "description": (
                "Agent 音频原料索引，镜像 Mongo somni_audio_materials；"
                "含六维嵌套标签，检索读路径使用"
            ),
            "source_collection": "somni_audio_materials",
            "field_descriptions": {
                "audio_name": "音频名称，用于后台管理、展示与检索出参",
                "description": "音频描述，说明内容、声音特点或适用场景",
                "status": "是否启用；true 表示可参与检索与同步",
                "audio_url": "音频文件 CDN / 对象存储 URL",
                "operation_type": "0 大模型打标，1 人工打标",
                "tag_id": "标签 ID，关联 somni_audio_tag_dictionary._id",
                "code": "标签编码，用于规则判断与接口传输",
                "name": "标签中文名，用于展示与检索精确匹配",
                "en_name": "标签英文名（原料表字段 en_name）",
                "parent_tag_id": "父标签 ID，二级分类指向一级",
                "parent_tag_code": "父标签编码",
                "value": "工程特征取值子对象，如 event_density=low",
                "band_values": "各频段占比（spectral_profile）",
                "relative_loudness": "相对响度（spectral_profile）",
                "sleep_stage_tags": (
                    "睡眠阶段（多选）：unwind/soothe/guard/wake"
                ),
                "content_form_tags": (
                    "内容形态标签（多选）：一级/二级分类，如 music → slow_piano"
                ),
                "mechanism_tags": "作用机制标签（多选）：如降低唤醒、声音遮蔽、注意力锚定",
                "audio_engineering_tags": (
                    "音频工程特征：维度 code + value 取值；spectral_profile 含 band_values、relative_loudness"
                ),
                "medical_risk_tags": "医学风控标签（多选）",
                "evidence_level_tags": "证据等级 A/B/C/D/R/X，用于精排权重",
            },
        },
        "properties": {
            "audio_name": _field("keyword", "音频名称，用于后台管理、展示与检索出参"),
            "description": _field(
                "text",
                "音频描述，说明内容、声音特点或适用场景（通常为大模型生成）",
            ),
            "status": _field("boolean", "是否启用；true 表示可参与检索与同步"),
            "audio_url": _field("keyword", "音频文件地址（CDN / 对象存储 / 内部资源 URL）"),
            "operation_type": _field(
                "integer",
                "标注操作类型：0 大模型打标，1 人工打标",
            ),
            "created_by": _field("keyword", "创建人"),
            "updated_by": _field("keyword", "最后更新人"),
            "created_at": _field("date", "记录创建时间"),
            "updated_at": _field("date", "记录最后更新时间；同步 diff 依据之一"),
            "sleep_stage_tags": _nested(tag_ref),
            "content_form_tags": _nested(
                {
                    **tag_ref,
                    "en_name": _field("keyword", "标签英文名称（原料表字段名 en_name）"),
                    "parent_tag_id": _field("keyword", "父标签 ID；一级分类为空，二级指向一级"),
                    "parent_tag_code": _field("keyword", "父标签编码；与 parent_tag_id 对应"),
                }
            ),
            "mechanism_tags": _nested(tag_ref),
            "audio_engineering_tags": _nested(
                {
                    **tag_ref,
                    "value": {"type": "object", "properties": tag_ref},
                    "band_values": _field(
                        "float",
                        "各频段占比或强度值数组（spectral_profile 专用）",
                    ),
                    "relative_loudness": _field(
                        "float",
                        "相对响度（spectral_profile 专用，如 integrated LUFS）",
                    ),
                }
            ),
            "medical_risk_tags": _nested(tag_ref),
            "evidence_level_tags": _nested(tag_ref),
        },
    }


def build_somni_audio_tag_dictionary_mapping(embedding_dim: int) -> dict[str, Any]:
    """somni_audio_tag_dictionary 索引 mapping（Mongo 词典 + 向量字段）。"""
    return {
        "_meta": {
            "description": (
                "六维标签字典索引，镜像 Mongo somni_audio_tag_dictionary；"
                "同步时为 name / name_en 生成向量供模糊检索"
            ),
            "source_collection": "somni_audio_tag_dictionary",
            "field_descriptions": {
                "type": "标签维度：sleep_stage/content_form/mechanism 等",
                "code": "标签编码，用于查询与规则判断",
                "status": "标签状态，同步仅纳入「启用」",
                "name": "标签中文名",
                "name_en": "标签英文名",
                "name_vector": f"name 的 {embedding_dim} 维 embedding，供模糊检索",
                "name_en_vector": f"name_en 的 {embedding_dim} 维 embedding，空则零向量",
                "description": "按维度存放的说明对象（判定标准、睡眠建议等）",
                "applicability": "四阶段适用程度：unwind/soothe/guard/wake",
                "parent_tag_id": "父标签 ID，内容形态二级分类",
                "created_at": "创建时间",
                "updated_at": "更新时间",
                "created_by": "创建人",
                "updated_by": "更新人",
            },
        },
        "properties": {
            "type": _field(
                "keyword",
                "标签所属维度：sleep_stage / content_form / mechanism / "
                "audio_engineering / medical_risk / evidence_level",
            ),
            "code": _field("keyword", "标签编码，用于查询、规则判断与策略传输"),
            "status": _field("keyword", "标签状态；同步仅纳入「启用」"),
            "name": _field("keyword", "标签中文名称，用于后台展示与人工标注"),
            "name_en": _field("keyword", "标签英文名称，用于多语言展示"),
            "name_vector": _field(
                "dense_vector",
                f"name 的 embedding 向量（{embedding_dim} 维，bge-small-zh-v1.5），供检索模糊匹配",
                dims=embedding_dim,
            ),
            "name_en_vector": _field(
                "dense_vector",
                f"name_en 的 embedding 向量（{embedding_dim} 维）；name_en 为空时存零向量",
                dims=embedding_dim,
            ),
            "description": {"type": "object", "enabled": True},
            "applicability": {"type": "object", "enabled": True},
            "parent_tag_id": _field("keyword", "父标签 ID，用于内容形态二级分类"),
            "created_at": _field("date", "标签创建时间"),
            "updated_at": _field("date", "标签最后更新时间；同步 diff 依据之一"),
            "created_by": _field("keyword", "标签创建人"),
            "updated_by": _field("keyword", "标签最后更新人"),
        },
    }
