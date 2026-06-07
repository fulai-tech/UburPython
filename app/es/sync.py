"""Elasticsearch 写路径（EsSync）。

CUD 经 comm 写 Mongo 后，由本模块同步 ES：扁平 tags → 六维结构 + embedding → upsert。
Mongo 侧 tags 仍为 string[] 时，靠 TAG_DIMENSION_PREFIXES 拆维（待业务定稿后可改为配置表）。
"""

from __future__ import annotations

import uuid

from elasticsearch import AsyncElasticsearch
from loguru import logger

from app.core.config import Settings
from app.core.tags import resolve_flat_tag
from app.embedding.encoder import Encoder
from app.es.search import EsSearch


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
        return resolve_flat_tag(tag)

    async def _embed_and_store_tag(self, dimension: str, label: str) -> str:
        """标签 embedding 写入 tag_vectors；同名 label 已存在则复用 _id。"""
        existing_id = await self._search.find_tag_vector_id_by_label(label)
        if existing_id:
            logger.debug("复用已有标签向量，label={}，id={}", label, existing_id)
            return existing_id

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
        """仅删除 audio_materials；tag_vectors 保留供其他音频复用。"""
        try:
            await self._client.delete(index=self.audio_index, id=doc_id)
            logger.info("ES 已删除音频索引文档，id={}（tag_vectors 未动）", doc_id)
        except Exception as exc:
            logger.warning("ES 删除音频失败，id={}，原因：{}", doc_id, exc)
