"""Elasticsearch 写路径（EsSync）。

CUD 经 comm 写 Mongo 后，由本模块同步 ES：扁平 tags → 六维结构 + embedding → upsert。
Mongo 侧 tags 仍为 string[] 时，靠 TAG_DIMENSION_PREFIXES 拆维（待业务定稿后可改为配置表）。
"""

from __future__ import annotations

import uuid
from typing import Any

from elasticsearch import AsyncElasticsearch
from loguru import logger

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.search import EsSearch

# 扁平 tag 前缀 → ES 六维字段名；无前缀默认归入 content_form
TAG_DIMENSION_PREFIXES: dict[str, str] = {
    "sleep:": "sleep_stage",
    "content:": "content_form",
    "mechanism:": "mechanism",
    "feat:": "audio_feat",
    "rhythm:": "rhythm",
    "risk:": "risk_control",
}

DEFAULT_DIMENSION = "content_form"


class EsSync:
    """写路径：六维拆分、embed、upsert/delete ES 索引。"""

    def __init__(
        self,
        client: AsyncElasticsearch,
        encoder: Encoder,
        settings: Settings,
    ) -> None:
        self._client = client
        self._encoder = encoder
        self._settings = settings
        self._search = EsSearch(client, settings)

    @property
    def audio_index(self) -> str:
        return self._settings.es_audio_index

    @property
    def tag_vectors_index(self) -> str:
        return self._settings.es_tag_vectors_index

    def _resolve_dimension(self, tag: str) -> tuple[str, str]:
        for prefix, dimension in TAG_DIMENSION_PREFIXES.items():
            if tag.startswith(prefix):
                return dimension, tag[len(prefix) :]
        return DEFAULT_DIMENSION, tag

    async def _embed_and_store_tag(self, dimension: str, label: str) -> str:
        """每个标签独立 embedding 并写入 tag_vectors；_id 模拟 ObjectId 长度。"""
        vector_id = uuid.uuid4().hex[:24]
        vector = await self._encoder.encode_one(label)
        await self._client.index(
            index=self.tag_vectors_index,
            id=vector_id,
            document={
                "label_text": label,
                "dimension": dimension,
                "vector": vector,
            },
        )
        return vector_id

    async def _build_tags_structure(self, flat_tags: list[str]) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = {
            "sleep_stage": [],
            "content_form": [],
            "mechanism": [],
            "audio_feat": [],
            "rhythm": [],
            "risk_control": [],
        }
        for tag in flat_tags:
            dimension, label = self._resolve_dimension(tag)
            vector_id = await self._embed_and_store_tag(dimension, label)
            grouped[dimension].append({"vector_id": vector_id, "label": label})
        return grouped

    async def upsert_audio(
        self,
        doc_id: str,
        *,
        audio_url: str,
        audio_name: str,
        flat_tags: list[str],
        evidence_level: str,
        recommend_weight: float,
    ) -> None:
        """doc_id 与 comm Mongo _id 保持一致，便于对账。"""
        tags = await self._build_tags_structure(flat_tags)
        document = {
            "audio_url": audio_url,
            "audio_name": audio_name,
            "tags": tags,
            "evidence_level": evidence_level,
            "recommend_weight": recommend_weight,
        }
        await self._client.index(index=self.audio_index, id=doc_id, document=document)
        logger.info("ES 已写入音频索引，id={}，名称={}", doc_id, audio_name)

    async def delete_audio(self, doc_id: str) -> None:
        # ES 文档可能已被手动删或从未同步，删失败只 warn 不阻断 comm 删 Mongo
        try:
            await self._client.delete(index=self.audio_index, id=doc_id)
            logger.info("ES 已删除音频，id={}", doc_id)
        except Exception as exc:
            logger.warning("ES 删除音频失败，id={}，原因：{}", doc_id, exc)

    async def delete_tag_vectors_for_doc(self, doc: dict[str, Any]) -> None:
        """可选清理：删 audio 时回收孤立 tag_vectors（当前 delete 路径未默认调用）。"""
        tags = doc.get("tags", {})
        vector_ids: list[str] = []
        for dim_items in tags.values():
            if isinstance(dim_items, list):
                for item in dim_items:
                    if vid := item.get("vector_id"):
                        vector_ids.append(vid)
        for vid in vector_ids:
            try:
                await self._client.delete(index=self.tag_vectors_index, id=vid)
            except Exception as exc:
                logger.warning("ES 删除标签向量失败，id={}，原因：{}", vid, exc)
