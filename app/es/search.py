"""Elasticsearch 读路径。

索引：somni_audio_materials（音频原料）、somni_audio_tag_dictionary（标签词典 + 向量）。
/search 只读此模块，不写 ES（规范 §九）。
"""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError
from loguru import logger

from app.core.config import Settings
from app.es.index_mappings import (
    build_somni_audio_materials_mapping,
    build_somni_audio_tag_dictionary_mapping,
)
from app.schemas.audio import AudioTags, TagItem

LEGACY_INDICES = ("audio_materials", "tag_vectors")


def _tag_item_from_dict(item: dict[str, Any]) -> TagItem | None:
    tag_id = str(item.get("tag_id", "")).strip()
    label = str(item.get("name", "")).strip()
    if not tag_id or not label:
        return None
    return TagItem(vector_id=tag_id, label=label)


def _parse_tag_list(items: list[dict[str, Any]] | None) -> list[TagItem]:
    if not items:
        return []
    parsed: list[TagItem] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tag = _tag_item_from_dict(item)
        if tag:
            parsed.append(tag)
    return parsed


def _parse_engineering_tags(items: list[dict[str, Any]] | None) -> list[TagItem]:
    if not items:
        return []
    parsed: list[TagItem] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tag = _tag_item_from_dict(item)
        if tag:
            parsed.append(tag)
    return parsed


def _document_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """ES hit → 索引文档（id + _source，不做字段裁剪）。"""
    source = hit.get("_source", {})
    if not isinstance(source, dict):
        return {"id": hit.get("_id", "")}
    return {"id": hit["_id"], **source}


class EsSearch:
    """封装检索相关的 ES 查询与文档解析。"""

    def __init__(self, client: AsyncElasticsearch, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    @property
    def audio_index(self) -> str:
        return self._settings.es_audio_index

    @property
    def tag_dictionary_index(self) -> str:
        return self._settings.es_tag_vectors_index

    @property
    def tag_vectors_index(self) -> str:
        return self.tag_dictionary_index

    async def filter_by_sleep_stage(self, sleep_stage_tags: list[str]) -> list[dict[str, Any]]:
        """检索步骤 1：睡眠阶段 nested term 精确匹配。"""
        if not sleep_stage_tags:
            return []

        should = [
            {
                "nested": {
                    "path": "sleep_stage_tags",
                    "query": {"term": {"sleep_stage_tags.name": tag}},
                }
            }
            for tag in sleep_stage_tags
        ]
        query = {"query": {"bool": {"should": should, "minimum_should_match": 1}}, "size": 1000}
        response = await self._client.search(index=self.audio_index, body=query)
        return [_document_from_hit(hit) for hit in response["hits"]["hits"]]

    async def list_all_audio_candidates(self) -> list[dict[str, Any]]:
        """检索步骤 1（跳过睡眠阶段过滤时）：返回索引内全部音频候选。"""
        query = {"query": {"match_all": {}}, "size": 1000}
        response = await self._client.search(index=self.audio_index, body=query)
        return [_document_from_hit(hit) for hit in response["hits"]["hits"]]

    async def find_tag_doc_id_by_name(self, name: str) -> str | None:
        """按标签中文名查词典索引，命中则返回文档 _id。"""
        if not name:
            return None
        response = await self._client.search(
            index=self.tag_dictionary_index,
            body={"query": {"term": {"name": name}}, "size": 1, "_source": False},
        )
        hits = response["hits"]["hits"]
        return hits[0]["_id"] if hits else None

    async def find_tag_vector_id_by_label(self, label: str) -> str | None:
        return await self.find_tag_doc_id_by_name(label)

    async def list_all_audio_doc_ids(self) -> set[str]:
        """音频索引全部 _id（用于与源库对账删孤儿）。"""
        return await self._list_all_doc_ids(self.audio_index)

    async def list_all_tag_dictionary_doc_ids(self) -> set[str]:
        """标签词典索引全部 _id。"""
        return await self._list_all_doc_ids(self.tag_dictionary_index)

    async def _list_all_doc_ids(self, index: str) -> set[str]:
        doc_ids: set[str] = set()
        search_after: list[str] | None = None
        while True:
            body: dict[str, Any] = {
                "query": {"match_all": {}},
                "_source": False,
                "size": 500,
                "sort": ["_doc"],
            }
            if search_after is not None:
                body["search_after"] = search_after
            response = await self._client.search(index=index, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                doc_ids.add(hit["_id"])
            search_after = hits[-1]["sort"]
        return doc_ids

    async def get_audio_source(self, doc_id: str) -> dict[str, Any] | None:
        """按 _id 取音频文档 _source；不存在返回 None。"""
        try:
            response = await self._client.get(index=self.audio_index, id=doc_id)
            source = response.get("_source")
            return source if isinstance(source, dict) else None
        except NotFoundError:
            return None

    async def get_tag_dictionary_source(self, doc_id: str) -> dict[str, Any] | None:
        try:
            response = await self._client.get(index=self.tag_dictionary_index, id=doc_id)
            source = response.get("_source")
            return source if isinstance(source, dict) else None
        except NotFoundError:
            return None

    async def get_dictionary_vectors(self, tag_ids: list[str]) -> dict[str, list[float]]:
        """批量取标签词典 name_vector，供内容形态向量模糊命中。"""
        if not tag_ids:
            return {}
        response = await self._client.mget(index=self.tag_dictionary_index, body={"ids": tag_ids})
        result: dict[str, list[float]] = {}
        for doc in response["docs"]:
            if doc.get("found"):
                result[doc["_id"]] = doc["_source"].get("name_vector", [])
        return result

    async def get_tag_vectors(self, vector_ids: list[str]) -> dict[str, list[float]]:
        return await self.get_dictionary_vectors(vector_ids)

    @staticmethod
    def parse_tags(raw: dict[str, Any]) -> AudioTags:
        """ES somni_audio_materials 文档 → Pydantic AudioTags。"""
        return AudioTags(
            sleep_stage=_parse_tag_list(raw.get("sleep_stage_tags")),
            content_form=_parse_tag_list(raw.get("content_form_tags")),
            mechanism=_parse_tag_list(raw.get("mechanism_tags")),
            audio_feat=_parse_engineering_tags(raw.get("audio_engineering_tags")),
            rhythm=[],
            risk_control=_parse_tag_list(raw.get("medical_risk_tags")),
        )

    @staticmethod
    def content_tag_ids(tags: AudioTags) -> list[str]:
        """内容准入/厌恶剔除用的 tag_id 列表（content_form + mechanism + audio_feat）。"""
        ids: list[str] = []
        for dim in (tags.content_form, tags.mechanism, tags.audio_feat):
            ids.extend(item.vector_id for item in dim)
        return ids

    async def migrate_legacy_indices(self) -> None:
        """删除旧版 audio_materials / tag_vectors 索引。"""
        for index in LEGACY_INDICES:
            if await self._client.indices.exists(index=index):
                await self._client.indices.delete(index=index)
                logger.info("已删除旧版 ES 索引：{}", index)

    async def ensure_indices(self) -> None:
        """启动时幂等建索引；dims 必须与 EMBEDDING_DIM 一致。"""
        dim = self._settings.embedding_dim

        materials_mapping = {"mappings": build_somni_audio_materials_mapping(dim)}
        dictionary_mapping = {"mappings": build_somni_audio_tag_dictionary_mapping(dim)}

        for index, mapping in (
            (self.audio_index, materials_mapping),
            (self.tag_dictionary_index, dictionary_mapping),
        ):
            if not await self._client.indices.exists(index=index):
                await self._client.indices.create(index=index, body=mapping)
                logger.info("已创建 ES 索引：{}", index)
