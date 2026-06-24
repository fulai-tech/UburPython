"""Elasticsearch 写路径（EsSync）。

Somni 数据以 Mongo 同步为准；HTTP CUD 经 comm 写 Mongo 后不再 upsert ES（避免破坏新索引结构）。
删除操作仍从 somni_audio_materials 索引移除对应文档。
"""

from __future__ import annotations

from elasticsearch import AsyncElasticsearch, NotFoundError
from loguru import logger

from app.core.config import Settings
from app.embedding.encoder import Encoder


class EsSync:
    """写路径：HTTP CUD 触发的 ES 删除；upsert 由 Mongo 同步脚本维护。"""

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
        """Somni 索引由 Mongo 同步维护，HTTP CUD 跳过 ES upsert。"""
        logger.warning(
            "HTTP CUD 跳过 ES upsert（id={}，name={}），请使用 Mongo 同步脚本写入 Somni 索引",
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
