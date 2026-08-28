"""Elasticsearch 读路径。

索引：somni_audio_materials（音频原料）、somni_audio_tag_dictionary（标签词典 + 向量）。
/search 只读此模块，不写 ES（规范 §九）。
"""

from __future__ import annotations

import asyncio
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
CONTENT_TAG_TYPES = ("content_form", "mechanism", "audio_engineering")
# 检索候选只取流水线与响应组装需要的字段，避免新增索引字段被默认带回。
SEARCH_CANDIDATE_SOURCE_INCLUDES = (
    "audio_name",
    "description",
    "audio_url",
    "cover_url",
    "sleep_stage_tags",
    "content_form_tags",
    "mechanism_tags",
    "audio_engineering_tags",
    "medical_risk_tags",
    "evidence_level_tags",
    "recommend_weight",
)


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
    """ES hit → 候选文档（只注入 hit._id，避免与 _source id 混淆）。"""
    source = hit.get("_source", {})
    doc_id = str(hit.get("_id", ""))
    if not isinstance(source, dict):
        return {"_id": doc_id}
    return {"_id": doc_id, **source}


def _candidate_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """检索候选：_source 已按 includes 裁剪，附带 hit._id 作为 _id。"""
    return _document_from_hit(hit)


def _candidate_search_body(query: dict[str, Any], *, size: int = 1000) -> dict[str, Any]:
    """检索候选查询体：只取流水线必要字段。"""
    return {
        "query": query,
        "size": size,
        "_source": {"includes": list(SEARCH_CANDIDATE_SOURCE_INCLUDES)},
    }


class EsSearch:
    """封装检索相关的 ES 查询与文档解析。"""

    def __init__(
        self,
        client: AsyncElasticsearch,
        settings: Settings,
        *,
        audio_index: str | None = None,
        tag_dictionary_index: str | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._audio_index = audio_index or settings.es_audio_index
        self._tag_dictionary_index = (
            tag_dictionary_index or settings.es_tag_vectors_index
        )
        self._content_tag_vectors_cache: list[dict[str, Any]] | None = None
        self._content_tag_vectors_lock = asyncio.Lock()
        # 按 tag_id 缓存 name_vector，避免每请求 mget（内容准入模糊路径）
        self._dictionary_vectors_cache: dict[str, list[float]] = {}
        self._dictionary_vectors_lock = asyncio.Lock()

    @property
    def audio_index(self) -> str:
        return self._audio_index

    @property
    def tag_dictionary_index(self) -> str:
        return self._tag_dictionary_index

    @property
    def tag_vectors_index(self) -> str:
        return self.tag_dictionary_index

    async def filter_by_sleep_stage(self, sleep_stage_tags: list[str]) -> list[dict[str, Any]]:
        """检索步骤 1：sleep_stage_names term 精确匹配。"""
        if not sleep_stage_tags:
            return []

        response = await self._client.search(
            index=self.audio_index,
            body=_candidate_search_body(_sleep_stage_filter(sleep_stage_tags)),
        )
        return [_candidate_from_hit(hit) for hit in response["hits"]["hits"]]

    async def list_all_audio_candidates(self) -> list[dict[str, Any]]:
        """检索步骤 1（跳过睡眠阶段过滤时）：返回索引内全部音频候选。"""
        response = await self._client.search(
            index=self.audio_index,
            body=_candidate_search_body({"match_all": {}}),
        )
        return [_candidate_from_hit(hit) for hit in response["hits"]["hits"]]

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
            if isinstance(source, dict):
                return {"id": doc_id, "_id": doc_id, **source}
            return None
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
        """去重读取标签词典 name_vector；进程内缓存命中则跳过 ES mget。"""
        unique_tag_ids = list(dict.fromkeys(tag_id for tag_id in tag_ids if tag_id))
        if not unique_tag_ids:
            return {}

        cached = {
            tag_id: self._dictionary_vectors_cache[tag_id]
            for tag_id in unique_tag_ids
            if tag_id in self._dictionary_vectors_cache
        }
        missing = [tag_id for tag_id in unique_tag_ids if tag_id not in cached]
        if not missing:
            return cached

        async with self._dictionary_vectors_lock:
            cached = {
                tag_id: self._dictionary_vectors_cache[tag_id]
                for tag_id in unique_tag_ids
                if tag_id in self._dictionary_vectors_cache
            }
            missing = [tag_id for tag_id in unique_tag_ids if tag_id not in cached]
            if missing:
                fetched = await self._mget_dictionary_vectors(missing)
                self._dictionary_vectors_cache.update(fetched)
                cached.update(fetched)

        return {
            tag_id: cached[tag_id] for tag_id in unique_tag_ids if tag_id in cached
        }

    async def _mget_dictionary_vectors(self, tag_ids: list[str]) -> dict[str, list[float]]:
        """分批 mget 标签词典 name_vector（仅未命中缓存的 id）。"""
        batch_size = max(1, self._settings.es_dictionary_mget_batch_size)
        result: dict[str, list[float]] = {}
        batch_count = 0
        for offset in range(0, len(tag_ids), batch_size):
            batch_count += 1
            batch = tag_ids[offset : offset + batch_size]
            response = await self._client.mget(
                index=self.tag_dictionary_index,
                ids=batch,
                source_includes=["name_vector"],
            )
            for doc in response["docs"]:
                if doc.get("found"):
                    result[doc["_id"]] = doc.get("_source", {}).get("name_vector", [])

        logger.debug(
            "标签词典向量批量读取：请求数={}，批次数={}，命中向量数={}",
            len(tag_ids),
            batch_count,
            len(result),
        )
        return result

    async def get_tag_vectors(self, vector_ids: list[str]) -> dict[str, list[float]]:
        return await self.get_dictionary_vectors(vector_ids)

    async def list_content_tag_vectors(self, *, size: int = 1000) -> list[dict[str, Any]]:
        """读取内容相关标签词典向量；进程内缓存，避免每请求打 ES。"""
        if self._content_tag_vectors_cache is not None:
            return self._content_tag_vectors_cache

        async with self._content_tag_vectors_lock:
            if self._content_tag_vectors_cache is not None:
                return self._content_tag_vectors_cache
            tags = await self._fetch_content_tag_vectors(size=size)
            self._content_tag_vectors_cache = tags
            async with self._dictionary_vectors_lock:
                self._seed_dictionary_vectors_from_content_tags(tags)
            logger.info("已缓存内容标签词典向量，数量={}", len(tags))
            return tags

    async def warm_dictionary_vectors_cache(self, *, size: int = 1000) -> None:
        """启动预热：拉取内容标签词典并写入按 id 向量缓存。"""
        tags = await self.list_content_tag_vectors(size=size)
        logger.info("标签词典向量缓存预热完成，数量={}", len(tags))

    def _seed_dictionary_vectors_from_content_tags(
        self,
        tags: list[dict[str, Any]],
    ) -> None:
        """把 list_content_tag_vectors 结果写入按 id 缓存。"""
        for tag in tags:
            tag_id = str(tag.get("id", "")).strip()
            vector = tag.get("vector")
            if tag_id and isinstance(vector, list) and vector:
                self._dictionary_vectors_cache[tag_id] = vector

    def clear_content_tag_vectors_cache(self) -> None:
        """词典同步后失效缓存（内容列表 + 按 id 向量）。"""
        self._content_tag_vectors_cache = None
        self._dictionary_vectors_cache.clear()

    async def _fetch_content_tag_vectors(self, *, size: int) -> list[dict[str, Any]]:
        response = await self._client.search(
            index=self.tag_dictionary_index,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"type": list(CONTENT_TAG_TYPES)}},
                            {"term": {"status": "启用"}},
                        ]
                    }
                },
                "size": size,
            },
        )
        tags: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source") or {}
            label = source.get("name")
            vector = source.get("name_vector")
            vector_en = source.get("name_en_vector")
            if not label:
                continue
            if not vector and not vector_en:
                continue
            tags.append(
                {
                    "id": hit.get("_id", ""),
                    "label": label,
                    "name_en": str(source.get("name_en") or ""),
                    "code": str(source.get("code") or ""),
                    "dimension": source.get("type", ""),
                    "vector": vector or [],
                    "vector_en": vector_en or [],
                    "parent_tag_id": str(source.get("parent_tag_id") or ""),
                }
            )
        return tags

    async def search_by_description_vector(
        self,
        query_vector: list[float],
        *,
        sleep_stage_tags: list[str],
        size: int,
    ) -> list[dict[str, Any]]:
        """description_vector 语义召回；缺少向量的旧文档自然不会命中。"""
        if not query_vector or size <= 0:
            return []

        knn: dict[str, Any] = {
            "field": "description_vector",
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size * 3, 50),
        }
        if sleep_stage_tags:
            knn["filter"] = _sleep_stage_filter(sleep_stage_tags)

        response = await self._client.search(
            index=self.audio_index,
            body={
                "knn": knn,
                "size": size,
                "_source": {"includes": list(SEARCH_CANDIDATE_SOURCE_INCLUDES)},
            },
        )
        results: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            source = _candidate_from_hit(hit)
            source["_description_score"] = float(hit.get("_score") or 0.0)
            results.append(source)
        return results

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

    async def list_audio_catalog_docs(
        self,
        *,
        size: int,
        language: str,
    ) -> list[dict[str, Any]]:
        """量产 GetAudio：按 language 拉取音频（不含 embedding），供内存过滤。"""
        response = await self._client.search(
            index=self.audio_index,
            body={
                "query": {"term": {"language": language}},
                "size": max(1, size),
                "_source": {"excludes": ["embedding", "description_vector"]},
            },
        )
        return [_document_from_hit(hit) for hit in response["hits"]["hits"]]

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
            else:
                await self._client.indices.put_mapping(index=index, body=mapping["mappings"])


def _sleep_stage_filter(sleep_stage_tags: list[str]) -> dict[str, Any]:
    """扁平 sleep_stage_names 上的 terms 过滤（任一阶段命中即可）。"""
    return {"terms": {"sleep_stage_names": sleep_stage_tags}}
