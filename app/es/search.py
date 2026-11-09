"""Elasticsearch 读路径。

索引：somni_audio_materials（音频原料）、somni_audio_tag_dictionary（标签词典 + 向量）。
/search 只读此模块，不写 ES（规范 §九）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from elasticsearch import ApiError, AsyncElasticsearch, NotFoundError
from loguru import logger

from app.core.audio_tags import canonical_tag, canonical_tags, reference_tag
from app.core.config import Settings
from app.es.index_mappings import (
    build_handboard_audio_materials_mapping,
    build_handboard_audio_tag_dictionary_mapping,
    build_somni_audio_materials_mapping,
    build_somni_audio_tag_dictionary_mapping,
)
from app.es.somni_docs import description_text_field, description_vector_field
from app.schemas.audio import AudioTags, TagItem
from app.server.somni.audio.bm25_query import build_bm25_must_clause

LEGACY_INDICES = ("audio_materials", "tag_vectors")
GET_AUDIO_SOURCE_INCLUDES = (
    "audio_name",
    "audio_name_en",
    "short_description",
    "short_description_en",
    "audio_url",
    "cover_url",
    "content_form_tags",
    "vip",
    "language",
    "status",
    "created_at",
    "acoustic_metrics",
)
_GET_AUDIO_BM25_SOURCE = GET_AUDIO_SOURCE_INCLUDES
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
    "evidence_level_tags",
    "duration_seconds",
    "acoustic_metrics",
    "language",
    "languages",
    "source",
    "short_description",
    "recommend_weight",
    "status",
    "audio_name_en",
    "short_description_en",
)


def _tag_item_from_dict(item: dict[str, Any]) -> TagItem | None:
    tag_id = str(item.get("tag_id", "")).strip()
    label = reference_tag(item) or canonical_tag(item.get("name"), strict=False)
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


def build_get_audio_filters(
    *,
    tag_code: str = "",
    content_form_codes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """GetAudio 两路检索共用过滤：启用、可选内容形态 code（不再按请求 language 过滤）。"""
    filters: list[dict[str, Any]] = [
        {"term": {"status": True}},
    ]
    codes: list[str] = []
    single = (tag_code or "").strip()
    if single:
        codes = [single]
    elif content_form_codes:
        codes = [c.strip() for c in content_form_codes if c and str(c).strip()]
    if len(codes) == 1:
        filters.append(
            {
                "nested": {
                    "path": "content_form_tags",
                    "query": {"term": {"content_form_tags.code": codes[0]}},
                }
            }
        )
    elif len(codes) > 1:
        filters.append(
            {
                "nested": {
                    "path": "content_form_tags",
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"content_form_tags.code": code}} for code in codes
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                }
            }
        )
    return filters


def _scored_documents(response: dict[str, Any], *, score_field: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for hit in response.get("hits", {}).get("hits", []):
        doc = _document_from_hit(hit)
        doc[score_field] = float(hit.get("_score") or 0.0)
        docs.append(doc)
    return docs


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


CANDIDATE_QUERY_SIZE = 1000


def _candidate_search_body(
    query: dict[str, Any], *, size: int = CANDIDATE_QUERY_SIZE
) -> dict[str, Any]:
    """检索候选查询体：只取流水线必要字段。"""
    active_query = {
        "bool": {
            "filter": [{"term": {"status": True}}],
            "must": [query],
        }
    }
    return {
        "query": active_query,
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
        index_profile: Literal["handboard", "somni"] = "handboard",
    ) -> None:
        self._client = client
        self._settings = settings
        self._audio_index = audio_index or settings.es_audio_index
        self._tag_dictionary_index = tag_dictionary_index or settings.es_tag_vectors_index
        self._index_profile = index_profile
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
        results = [_candidate_from_hit(hit) for hit in response["hits"]["hits"]]
        if len(results) >= CANDIDATE_QUERY_SIZE:
            logger.warning(
                "音频候选达到单次查询上限 {}，超出部分未进入本次结果",
                CANDIDATE_QUERY_SIZE,
            )
        return results

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

        return {tag_id: cached[tag_id] for tag_id in unique_tag_ids if tag_id in cached}

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
                    "label": canonical_tag(source.get("code") or label),
                    "display_name": label,
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
        language: str = "",
        include_source: bool = True,
        allow_field_override: bool = True,
    ) -> list[dict[str, Any]]:
        """描述向量语义召回；按 language 选 zh/en 字段，不按文档 language 过滤。"""
        if not query_vector or size <= 0:
            return []

        vector_field = description_vector_field(
            language,
            override=self._settings.get_audio_description_vector_field
            if allow_field_override
            else "",
        )
        knn: dict[str, Any] = {
            "field": vector_field,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size * 3, 50),
        }
        filters: list[dict[str, Any]] = [
            {"term": {"status": True}},
            {"exists": {"field": vector_field}},
        ]
        if sleep_stage_tags:
            filters.append(_sleep_stage_filter(sleep_stage_tags))
        knn["filter"] = {"bool": {"filter": filters}}
        source: Any = (
            {"includes": list(SEARCH_CANDIDATE_SOURCE_INCLUDES)} if include_source else False
        )

        return await self._search_knn_documents(
            query_vector,
            vector_field=vector_field,
            k=size,
            num_candidates=max(size * 3, 50),
            filters=knn["filter"],
            source=source,
            score_field="_description_score",
        )

    async def _search_knn_documents(
        self,
        query_vector: list[float],
        *,
        vector_field: str,
        k: int,
        num_candidates: int,
        filters: Any,
        source: Any,
        score_field: str,
    ) -> list[dict[str, Any]]:
        """Shared ES vector request; each caller owns its filters/projection/score name."""
        response = await self._client.search(
            index=self.audio_index,
            body={
                "knn": {
                    "field": vector_field,
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": num_candidates,
                    "filter": filters,
                },
                "size": k,
                "_source": source,
            },
        )
        return _scored_documents(response, score_field=score_field)

    @staticmethod
    def parse_tags(raw: dict[str, Any]) -> AudioTags:
        """ES somni_audio_materials 文档 → Pydantic AudioTags。"""
        return AudioTags(
            sleep_stage=_parse_tag_list(raw.get("sleep_stage_tags")),
            content_form=_parse_tag_list(raw.get("content_form_tags")),
            mechanism=[],
            audio_feat=_parse_engineering_tags(raw.get("audio_engineering_tags")),
            rhythm=[],
            risk_control=[],
        )

    @staticmethod
    def content_tag_ids(tags: AudioTags) -> list[str]:
        """内容准入/厌恶剔除用的 tag_id 列表（content_form + mechanism + audio_feat）。"""
        ids: list[str] = []
        for dim in (tags.content_form, tags.mechanism, tags.audio_feat):
            ids.extend(item.vector_id for item in dim)
        return ids

    async def audio_index_exists(self) -> bool:
        return bool(await self._client.indices.exists(index=self.audio_index))

    async def refresh_audio_index(self) -> None:
        """让刚写入的音频立即可搜，避免紧接着的全量查询漏掉末尾几条。"""
        await self._client.indices.refresh(index=self.audio_index)

    async def list_audio_catalog_docs(
        self,
        *,
        size: int,
        language: str = "",
    ) -> list[dict[str, Any]]:
        """量产 GetAudio：拉取启用音频（不含向量），不再按请求 language 过滤。"""
        del language  # 保留入参兼容旧调用
        response = await self._client.search(
            index=self.audio_index,
            body={
                "query": {"term": {"status": True}},
                "size": max(1, size),
                "_source": {
                    "excludes": [
                        "embedding",
                        "description_vector",
                        "description_vector_en",
                    ]
                },
            },
        )
        return [_document_from_hit(hit) for hit in response["hits"]["hits"]]

    async def get_audio_docs_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """量产 GetAudioByIds：按文档 _id 批量取启用原料（不含向量）。"""
        unique_ids = [doc_id for doc_id in ids if doc_id]
        if not unique_ids:
            return []
        response = await self._client.search(
            index=self.audio_index,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"status": True}},
                            {"ids": {"values": unique_ids}},
                        ]
                    }
                },
                "size": len(unique_ids),
                "_source": {"includes": list(GET_AUDIO_SOURCE_INCLUDES)},
            },
        )
        return [_document_from_hit(hit) for hit in response["hits"]["hits"]]

    async def search_get_audio_bm25(
        self,
        query_text: str,
        *,
        language: str,
        tag_code: str = "",
        content_form_codes: list[str] | None = None,
        size: int,
        sleep_stage_tags: list[str] | None = None,
        source_includes: tuple[str, ...] = GET_AUDIO_SOURCE_INCLUDES,
    ) -> list[dict[str, Any]]:
        """量产 GetAudio：按 description_text / description_text_en 做 BM25。

        短词整句 match_phrase；长句拆成关键词 phrase（should）。
        """
        text_field = description_text_field(language)
        filters = build_get_audio_filters(tag_code=tag_code, content_form_codes=content_form_codes)
        if sleep_stage_tags:
            filters.append(_sleep_stage_filter(sleep_stage_tags))
        response = await self._client.search(
            index=self.audio_index,
            body={
                "size": max(1, size),
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [build_bm25_must_clause(text_field, query_text)],
                    }
                },
                "_source": {"includes": list(source_includes)},
            },
        )
        return _scored_documents(response, score_field="_bm25_score")

    async def search_get_audio_knn(
        self,
        query_vector: list[float],
        *,
        language: str,
        tag_code: str = "",
        content_form_codes: list[str] | None = None,
        k: int,
        num_candidates: int,
    ) -> list[dict[str, Any]]:
        """量产 GetAudio：按 description_vector / description_vector_en 做 Top-K。"""
        vector_field = description_vector_field(
            language,
            override=self._settings.get_audio_description_vector_field,
        )
        filters = build_get_audio_filters(
            tag_code=tag_code,
            content_form_codes=content_form_codes,
        )
        filters.append({"exists": {"field": vector_field}})
        return await self._search_knn_documents(
            query_vector,
            vector_field=vector_field,
            k=max(1, k),
            num_candidates=max(1, num_candidates),
            filters=filters,
            source={"includes": list(GET_AUDIO_SOURCE_INCLUDES)},
            score_field="_knn_score",
        )

    async def search_get_audio_vector_all(
        self,
        query_vector: list[float],
        *,
        language: str,
        tag_code: str = "",
        content_form_codes: list[str] | None = None,
        size: int,
    ) -> list[dict[str, Any]]:
        """量产 GetAudio：在过滤集上精确计算全部文档向量分数。"""
        vector_field = description_vector_field(
            language,
            override=self._settings.get_audio_description_vector_field,
        )
        filters = build_get_audio_filters(
            tag_code=tag_code,
            content_form_codes=content_form_codes,
        )
        filters.append({"exists": {"field": vector_field}})
        response = await self._client.search(
            index=self.audio_index,
            body={
                "size": max(1, size),
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": filters}},
                        "script": {
                            "source": (f"(cosineSimilarity(params.q, '{vector_field}') + 1.0) / 2"),
                            "params": {"q": query_vector},
                        },
                    }
                },
                "_source": {"includes": list(GET_AUDIO_SOURCE_INCLUDES)},
            },
        )
        return _scored_documents(response, score_field="_knn_score")

    async def migrate_legacy_indices(self) -> None:
        """删除旧版 audio_materials / tag_vectors 索引。"""
        for index in LEGACY_INDICES:
            if await self._client.indices.exists(index=index):
                await self._client.indices.delete(index=index)
                logger.info("已删除旧版 ES 索引：{}", index)

    async def ensure_indices(self) -> None:
        """幂等建索引：handboard / somni 两套 mapping 完全分开。

        已有索引走 put_mapping；dense_vector.dims 不可变，冲突时跳过并打日志。
        量产要换维须先删索引再 ensure（sync_es_from_somni 已处理标签索引）。
        """
        dim = self._settings.embedding_dim
        if self._index_profile == "somni":
            dim_en = self._settings.embedding_dim_en
            materials_mapping = {"mappings": build_somni_audio_materials_mapping(dim, dim_en)}
            dictionary_mapping = {"mappings": build_somni_audio_tag_dictionary_mapping(dim, dim_en)}
        else:
            materials_mapping = {"mappings": build_handboard_audio_materials_mapping(dim)}
            dictionary_mapping = {"mappings": build_handboard_audio_tag_dictionary_mapping(dim)}

        for index, mapping in (
            (self.audio_index, materials_mapping),
            (self.tag_dictionary_index, dictionary_mapping),
        ):
            if not await self._client.indices.exists(index=index):
                await self._client.indices.create(index=index, body=mapping)
                logger.info("已创建 ES 索引：{}", index)
                continue
            try:
                await self._client.indices.put_mapping(index=index, body=mapping["mappings"])
            except ApiError as exc:
                if not _is_dense_vector_dims_conflict(exc):
                    raise
                logger.warning(
                    "跳过更新 ES 映射（向量 dims 不可变，保留现有索引）：index={} err={}",
                    index,
                    exc.message,
                )


def _is_dense_vector_dims_conflict(exc: ApiError) -> bool:
    """put_mapping 因 dense_vector.dims 与现有字段冲突而失败。"""
    return "Cannot update parameter [dims]" in f"{exc.message} {exc}"


def _sleep_stage_filter(sleep_stage_tags: list[str]) -> dict[str, Any]:
    """按既有 nested sleep_stage_tags.code 过滤，任一英文阶段命中即可。"""
    return {
        "nested": {
            "path": "sleep_stage_tags",
            "query": {"terms": {"sleep_stage_tags.code": canonical_tags(sleep_stage_tags)}},
        }
    }
