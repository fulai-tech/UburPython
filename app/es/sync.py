"""Elasticsearch 写路径（EsSync）。

创建/更新：按 Somni 文档 upsert（补 description_text / description_vector）。
删除：从 somni_audio_materials 移除文档。
遗留 upsert_audio（扁平 tags）保留为空操作，避免旧调用破坏索引。
"""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError
from loguru import logger

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.somni_docs import material_source_for_es


class EsSync:
    """HTTP CUD 触发的 ES 写入与删除。"""

    def __init__(
        self,
        client: AsyncElasticsearch,
        encoder: Encoder,
        settings: Settings,
    ) -> None:
        self._client = client
        self._encoder = encoder
        self._settings = settings

    @property
    def audio_index(self) -> str:
        return self._settings.es_audio_index

    async def upsert_somni_material(self, doc_id: str, doc: dict[str, Any]) -> None:
        """写入 Somni 索引；无 audio_url 则跳过。"""
        payload = material_source_for_es(doc)
        if payload is None:
            logger.info("跳过 ES upsert：audio_url 缺失，id={}", doc_id)
            return
        zh_text = str(payload.get("description_text", "")).strip()
        en_text = str(payload.get("description_text_en", "")).strip()
        # cosine 字段不能写零向量，缺文本/缺模型时省略字段
        if zh_text:
            payload["description_vector"] = await self._encoder.encode_one(
                zh_text, language="zh"
            )
        else:
            payload.pop("description_vector", None)
        try:
            if en_text:
                payload["description_vector_en"] = await self._encoder.encode_one(
                    en_text, language="en"
                )
            else:
                payload.pop("description_vector_en", None)
        except FileNotFoundError:
            payload.pop("description_vector_en", None)
            logger.warning(
                "英文 ONNX 不可用，跳过 description_vector_en，id={}", doc_id
            )
        await self._client.index(index=self.audio_index, id=doc_id, document=payload)
        logger.info("ES 已 upsert 原料，id={}", doc_id)

    async def upsert_audio(
        self,
        doc_id: str,
        *,
        audio_url: str,
        audio_name: str,
        flat_tags: list[str],
        evidence_level: str,
        recommend_weight: float,
        description: str = "",
    ) -> None:
        """遗留扁平 upsert：空操作。"""
        logger.warning(
            "HTTP CUD 跳过遗留 ES upsert（id={}，name={}）",
            doc_id,
            audio_name,
        )

    async def delete_audio(self, doc_id: str) -> None:
        """删除 somni_audio_materials 索引文档。"""
        try:
            await self._client.delete(index=self.audio_index, id=doc_id)
            logger.info("ES 已删除音频索引文档，id={}", doc_id)
        except NotFoundError:
            logger.info("ES 删除跳过，文档不存在，id={}", doc_id)
        except Exception as exc:
            logger.warning("ES 删除音频失败，id={}，原因：{}", doc_id, exc)
