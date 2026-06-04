"""Elasticsearch 读路径。

索引：audio_materials（文档 + 六维标签）、tag_vectors（标签向量，供模糊准入）。
/search 只读此模块，不写 ES（规范 §九）。
"""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch
from loguru import logger

from app.core.config import Settings
from app.schemas.audio import AudioTags, TagItem


class EsSearch:
    """封装检索相关的 ES 查询与文档解析。"""

    def __init__(self, client: AsyncElasticsearch, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    @property
    def audio_index(self) -> str:
        return self._settings.es_audio_index

    @property
    def tag_vectors_index(self) -> str:
        return self._settings.es_tag_vectors_index

    async def filter_by_sleep_stage(self, sleep_stage_tags: list[str]) -> list[dict[str, Any]]:
        """检索步骤 1：睡眠阶段 term 精确匹配；无命中则流水线短路返回空。"""
        if not sleep_stage_tags:
            return []

        query = {
            "query": {
                "bool": {
                    "should": [
                        {"term": {"tags.sleep_stage.label": tag}} for tag in sleep_stage_tags
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": 1000,
        }
        response = await self._client.search(index=self.audio_index, body=query)
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def get_tag_vectors(self, vector_ids: list[str]) -> dict[str, list[float]]:
        """批量取 tag_vectors，供内容形态向量模糊命中（步骤 2）。"""
        if not vector_ids:
            return {}

        response = await self._client.mget(
            index=self.tag_vectors_index,
            body={"ids": vector_ids},
        )
        result: dict[str, list[float]] = {}
        for doc in response["docs"]:
            if doc.get("found"):
                result[doc["_id"]] = doc["_source"]["vector"]
        return result

    @staticmethod
    def parse_tags(raw: dict[str, Any]) -> AudioTags:
        """ES _source.tags → Pydantic AudioTags。"""

        def parse_dim(items: list[dict[str, Any]] | None) -> list[TagItem]:
            if not items:
                return []
            return [TagItem(vector_id=i["vector_id"], label=i["label"]) for i in items]

        tags = raw.get("tags", {})
        return AudioTags(
            sleep_stage=parse_dim(tags.get("sleep_stage")),
            content_form=parse_dim(tags.get("content_form")),
            mechanism=parse_dim(tags.get("mechanism")),
            audio_feat=parse_dim(tags.get("audio_feat")),
            rhythm=parse_dim(tags.get("rhythm")),
            risk_control=parse_dim(tags.get("risk_control")),
        )

    async def ensure_indices(self) -> None:
        """启动时幂等建索引；dims 必须与 EMBEDDING_DIM 一致。"""
        dim = self._settings.embedding_dim

        audio_mapping = {
            "mappings": {
                "properties": {
                    "audio_url": {"type": "keyword"},
                    "audio_name": {"type": "keyword"},
                    "evidence_level": {"type": "keyword"},
                    "recommend_weight": {"type": "float"},
                    "tags": {
                        "properties": {
                            dim_name: {
                                "properties": {
                                    "vector_id": {"type": "keyword"},
                                    "label": {"type": "keyword"},
                                }
                            }
                            for dim_name in (
                                "sleep_stage",
                                "content_form",
                                "mechanism",
                                "audio_feat",
                                "rhythm",
                                "risk_control",
                            )
                        }
                    },
                }
            }
        }

        tag_vectors_mapping = {
            "mappings": {
                "properties": {
                    "label_text": {"type": "keyword"},
                    "dimension": {"type": "keyword"},
                    "vector": {"type": "dense_vector", "dims": dim},
                }
            }
        }

        for index, mapping in (
            (self.audio_index, audio_mapping),
            (self.tag_vectors_index, tag_vectors_mapping),
        ):
            if not await self._client.indices.exists(index=index):
                await self._client.indices.create(index=index, body=mapping)
                logger.info("已创建 ES 索引：{}", index)
