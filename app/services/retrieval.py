"""三维度检索流水线（RetrievalService）。

顺序固定（规范 §五，不可调换）：
  1. 睡眠阶段精确过滤（ES term，无命中 → 空数组）
  2. 内容形态准入（精确交集 或 向量逐标签计分，余弦 ≥ SIM_THRESHOLD）
  3. 厌恶剔除 → 粗排（写入 tag_score = match_count）
  4. 精排（当前按 tag_score 降序截断；evidence_level 权重待业务完善）
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.schemas.audio import (
    EVIDENCE_WEIGHT_MAP,
    AudioTags,
    EvidenceLevel,
    SearchAudioRequest,
)
from app.schemas.search_material import project_search_material

if TYPE_CHECKING:
    from app.cache.sleep_stage_cache import SleepStageCandidateCache

AUTO_TAG_TOP_K = 5
AUTO_TAG_SIM_THRESHOLD = 0.62
AUTO_DISLIKE_SIM_THRESHOLD = 0.65
WEAK_DISLIKE_PENALTY = 0.2
VOICE_MARKER = "人声"
NO_VOICE_CODE = "none"
# 音工维名：几乎全库都有，进厌恶会自比对互杀；不参与自动/显式厌恶
IGNORED_DISLIKE_LABELS = frozenset({"声音事件密度"})
MIN_AUTO_TAG_LABEL_LEN = 2  # 过滤「低」「无」等过短子串误匹配
MAX_FALLBACK_TAG_LEN = 12  # 无词典映射时，超过此长度的自然语言句不参与硬剔除
NEGATIVE_MARKERS = ("不要", "避免", "讨厌", "不喜欢", "别")
MUTUALLY_EXCLUSIVE_CONTENT_TAGS = frozenset({"白噪音", "粉噪音", "棕噪音"})
COLOR_NOISE_PARENT_TAG = "颜色噪音"
# 颜色噪音别名 → 规范标签：归一后才能精确命中同色并触发兄弟色互斥；
# 英文别名以小写存储，匹配时统一转小写。
COLOR_NOISE_ALIASES: dict[str, str] = {
    "白噪音": "白噪音",
    "白噪": "白噪音",
    "白噪声": "白噪音",
    "白色噪音": "白噪音",
    "white noise": "白噪音",
    "粉噪音": "粉噪音",
    "粉噪": "粉噪音",
    "粉噪声": "粉噪音",
    "粉红噪音": "粉噪音",
    "粉红噪声": "粉噪音",
    "粉色噪音": "粉噪音",
    "pink noise": "粉噪音",
    "棕噪音": "棕噪音",
    "棕噪": "棕噪音",
    "棕噪声": "棕噪音",
    "棕色噪音": "棕噪音",
    "褐噪音": "棕噪音",
    "褐色噪音": "棕噪音",
    "布朗噪音": "棕噪音",
    "brown noise": "棕噪音",
}
DictionaryVectors = dict[str, list[float]]
TextVectorCache = OrderedDict[str, list[float]]


@dataclass
class ScoredCandidate:
    """流水线中间态：ES 文档 + 解析后的标签 + 粗排分数。"""

    source: dict[str, Any]
    tags: AudioTags
    match_count: int = 0
    tag_score: float = 0.0
    desc_score: float = 0.0
    final_score: float = 0.0
    dislike_penalty: float = 0.0
    evidence_level: EvidenceLevel = EvidenceLevel.C
    recommend_weight: float = 0.45


@dataclass(frozen=True)
class ExtractedQueryTags:
    content_tags: list[str]
    disliked_tags: list[str]


class RetrievalService:
    """三维度检索：睡眠阶段 → 内容形态 → 厌恶剔除 → 粗排 → 精排。"""

    def __init__(
        self,
        es_search: EsSearch,
        encoder: Encoder,
        settings: Settings,
        sleep_stage_cache: SleepStageCandidateCache | None = None,
    ) -> None:
        self._es_search = es_search
        self._encoder = encoder
        self._settings = settings
        self._sleep_stage_cache = sleep_stage_cache
        self._search_semaphore = asyncio.Semaphore(settings.search_max_concurrency)
        self._text_vector_cache: TextVectorCache = OrderedDict()
        self._text_vector_cache_max = max(1, settings.embedding_text_cache_size)

    async def search(self, request: SearchAudioRequest) -> list[dict[str, Any]]:
        async with self._search_semaphore:
            query_text = (request.query_text or "").strip()
            if query_text:
                return await self._search_text_multi_route(request, query_text)
            return await self._search_tag_only(request)

    async def _search_tag_only(self, request: SearchAudioRequest) -> list[dict[str, Any]]:
        top_k_label = request.top_k if request.top_k is not None else "全部"
        logger.info(
            "检索开始，睡眠阶段={}，内容标签={}，厌恶标签={}，top_k={}",
            request.sleep_stage_tags,
            request.content_tags,
            request.disliked_tags,
            top_k_label,
        )
        total_started = time.perf_counter()

        step1_started = time.perf_counter()
        candidates_raw = await self._fetch_step1_candidates(request.sleep_stage_tags)
        step1_ms = _elapsed_ms(step1_started)
        if not candidates_raw:
            if self._settings.search_sleep_stage_filter_enabled:
                logger.info("检索：睡眠阶段无匹配，短路返回空结果")
            _log_tag_pipeline_timing(step1_ms, 0.0, 0.0, 0.0, _elapsed_ms(total_started))
            return []

        step2_started = time.perf_counter()
        need_dictionary = bool(
            _strip_voice_mention_tags(_usable_dislike_tags(request.disliked_tags))
        ) or self._has_non_exact_content_candidate(
            candidates_raw,
            _normalize_color_noise_aliases(request.content_tags),
        )
        dictionary_vectors, content_tags, disliked_tags, content_vectors, dislike_vectors = (
            await self._prepare_content_admission_inputs(
                candidates_raw,
                request,
                need_dictionary=need_dictionary,
            )
        )
        admitted = await self._apply_content_admission(
            candidates_raw,
            content_tags,
            dictionary_vectors,
            request_vectors=content_vectors,
        )
        step2_ms = _elapsed_ms(step2_started)
        logger.info(
            "检索步骤2/4 内容形态准入：通过数={}，耗时={:.1f}毫秒",
            len(admitted),
            step2_ms,
        )
        if not admitted:
            logger.info("检索：内容形态准入无匹配，短路返回空结果")
            _log_tag_pipeline_timing(step1_ms, step2_ms, 0.0, 0.0, _elapsed_ms(total_started))
            return []

        step3_started = time.perf_counter()
        vector_disliked = _strip_voice_mention_tags(disliked_tags)
        filtered = await self._apply_dislike_filter(
            admitted,
            vector_disliked,
            dictionary_vectors,
            dislike_vectors=dislike_vectors,
        )
        filtered = _apply_voice_code_filter(
            filtered,
            disliked_tags=disliked_tags,
        )
        scored = self._apply_coarse_rank(filtered)
        step3_ms = _elapsed_ms(step3_started)
        logger.info(
            "检索步骤3/4 厌恶剔除+粗排：剩余数={}，耗时={:.1f}毫秒",
            len(scored),
            step3_ms,
        )

        step4_started = time.perf_counter()
        ranked = sorted(scored, key=lambda c: c.tag_score, reverse=True)
        if request.top_k is not None:
            ranked = ranked[: request.top_k]
        step4_ms = _elapsed_ms(step4_started)
        logger.info(
            "检索步骤4/4 精排截断：top_k={}，输出数={}，match_count序列={}，耗时={:.1f}毫秒",
            top_k_label,
            len(ranked),
            [c.match_count for c in ranked],
            step4_ms,
        )

        results = [project_search_material(c.source) for c in ranked]
        _log_tag_pipeline_timing(
            step1_ms,
            step2_ms,
            step3_ms,
            step4_ms,
            _elapsed_ms(total_started),
        )
        logger.info("检索完成，命中数={}", len(results))
        return results

    async def _search_text_multi_route(
        self,
        request: SearchAudioRequest,
        query_text: str,
    ) -> list[dict[str, Any]]:
        top_k_label = request.top_k if request.top_k is not None else "全部"
        total_started = time.perf_counter()

        parse_started = time.perf_counter()
        query_vector = await self._encode_one(query_text)
        tag_vectors = await self._es_search.list_content_tag_vectors()
        extracted = await self._extract_query_tags(query_text, query_vector, tag_vectors)
        explicit_content_tags = _normalize_color_noise_aliases(request.content_tags)
        content_tags = _unique_preserve_order([*explicit_content_tags, *extracted.content_tags])
        content_tags = _remove_mutually_exclusive_expansions(
            content_tags,
            preferred_labels=explicit_content_tags,
        )
        disliked_tags = _usable_dislike_tags(
            _normalize_to_dictionary_labels(
                [*request.disliked_tags, *extracted.disliked_tags],
                tag_vectors,
            )
        )
        vector_disliked_tags = _strip_voice_mention_tags(disliked_tags)
        content_tags = [tag for tag in content_tags if tag not in set(disliked_tags)]
        parse_ms = _elapsed_ms(parse_started)
        logger.info(
            "多路检索开始，query_text={}，显式内容标签={}，自动内容标签={}，"
            "显式厌恶标签={}，归一厌恶标签={}，top_k={}，解析耗时={:.1f}毫秒",
            query_text,
            request.content_tags,
            extracted.content_tags,
            request.disliked_tags,
            disliked_tags,
            top_k_label,
            parse_ms,
        )

        recall_started = time.perf_counter()
        desc_docs, tag_docs = await self._recall_text_routes(
            request=request,
            query_vector=query_vector,
            has_content_tags=bool(content_tags),
        )
        desc_docs = [
            doc
            for doc in desc_docs
            if not _has_mutually_exclusive_conflict(
                content_tags,
                self._es_search.parse_tags(doc).content_labels(),
            )
        ]
        recall_ms = _elapsed_ms(recall_started)
        desc_candidates = [
            self._candidate_from_doc(doc, desc_score=_parse_desc_score(doc)) for doc in desc_docs
        ]

        rank_started = time.perf_counter()
        vector_docs = [*tag_docs, *desc_docs] if vector_disliked_tags else tag_docs
        need_dictionary = bool(vector_disliked_tags) or self._has_non_exact_content_candidate(
            tag_docs,
            content_tags,
        )
        dict_task = asyncio.create_task(
            self._prefetch_dictionary_vectors(vector_docs, required=need_dictionary)
        )
        encode_task = asyncio.create_task(
            self._encode_request_tag_vectors(content_tags, vector_disliked_tags)
        )
        try:
            dictionary_vectors, (content_vectors, dislike_vectors) = await asyncio.gather(
                dict_task,
                encode_task,
            )
        except BaseException:
            dict_task.cancel()
            encode_task.cancel()
            await asyncio.gather(dict_task, encode_task, return_exceptions=True)
            raise

        tag_candidates: list[ScoredCandidate] = []
        if content_tags:
            tag_candidates = await self._score_content_candidates(
                tag_docs,
                content_tags,
                dictionary_vectors,
                request_vectors=content_vectors,
            )

        merged = await self._merge_and_rank_text_candidates(
            tag_candidates=tag_candidates,
            desc_candidates=desc_candidates,
            content_tags=content_tags,
            disliked_tags=vector_disliked_tags,
            voice_filter_tags=disliked_tags,
            dictionary_vectors=dictionary_vectors,
            dislike_vectors=dislike_vectors,
            top_k=request.top_k,
        )
        rank_ms = _elapsed_ms(rank_started)
        total_ms = _elapsed_ms(total_started)
        stages = {
            "查询解析": parse_ms,
            "双路召回": recall_ms,
            "打分融合": rank_ms,
        }
        slowest = max(stages.items(), key=lambda item: item[1])[0]
        logger.info(
            "多路检索完成，标签召回={}，描述召回={}，融合输出={}，final_score序列={}，"
            "耗时：解析={:.1f}毫秒，召回={:.1f}毫秒，打分融合={:.1f}毫秒，"
            "合计={:.1f}毫秒，最慢={}",
            len(tag_candidates),
            len(desc_candidates),
            len(merged),
            [round(c.final_score, 4) for c in merged],
            parse_ms,
            recall_ms,
            rank_ms,
            total_ms,
            slowest,
        )
        return [project_search_material(c.source) for c in merged]

    async def _fetch_step1_candidates(self, sleep_stage_tags: list[str]) -> list[dict[str, Any]]:
        """步骤 1：优先读睡眠阶段 Redis 缓存；未命中再查 ES。"""
        started = time.perf_counter()
        if not self._settings.search_sleep_stage_filter_enabled:
            candidates = await self._es_search.list_all_audio_candidates()
            logger.info(
                "检索步骤1/4 睡眠阶段过滤：已跳过，候选数={}，耗时={:.1f}毫秒",
                len(candidates),
                _elapsed_ms(started),
            )
            return candidates

        cached = await self._get_sleep_stage_cached(sleep_stage_tags)
        if cached is not None:
            logger.info(
                "检索步骤1/4 睡眠阶段过滤：缓存命中，候选数={}，耗时={:.1f}毫秒",
                len(cached),
                _elapsed_ms(started),
            )
            return cached

        candidates = await self._es_search.filter_by_sleep_stage(sleep_stage_tags)
        await self._backfill_sleep_stage_cache()
        logger.info(
            "检索步骤1/4 睡眠阶段过滤：候选数={}，耗时={:.1f}毫秒",
            len(candidates),
            _elapsed_ms(started),
        )
        return candidates

    async def _get_sleep_stage_cached(
        self,
        sleep_stage_tags: list[str],
    ) -> list[dict[str, Any]] | None:
        if self._sleep_stage_cache is None:
            return None
        try:
            return await self._sleep_stage_cache.get(sleep_stage_tags)
        except Exception as exc:
            logger.warning("读取睡眠阶段候选缓存失败，回退 ES：{}", exc)
            return None

    async def _backfill_sleep_stage_cache(self) -> None:
        if self._sleep_stage_cache is None:
            return
        try:
            await self._sleep_stage_cache.warm(self._load_sleep_stage_candidates)
        except Exception as exc:
            logger.warning("回填睡眠阶段候选缓存失败：{}", exc)

    async def _load_sleep_stage_candidates(self, stage: str) -> list[dict[str, Any]]:
        return await self._es_search.filter_by_sleep_stage([stage])

    async def warm_sleep_stage_cache(self) -> None:
        """启动或同步后预热四个阶段候选。"""
        await self._backfill_sleep_stage_cache()

    async def warm_query_tag_vectors(self) -> None:
        """启动预热：内容词典 label → 文本向量 LRU，消除首请求 ONNX 冷编码。"""
        started = time.perf_counter()
        tags = await self._es_search.list_content_tag_vectors()
        labels = _unique_preserve_order(
            [
                label
                for tag in tags
                if (label := str(tag.get("label", "")).strip())
            ]
        )
        if not labels:
            logger.info("查询标签向量缓存预热跳过：内容词典为空")
            return
        await self._encode_texts(labels)
        logger.info(
            "查询标签向量缓存预热完成，标签数={}，耗时={:.1f}毫秒",
            len(labels),
            _elapsed_ms(started),
        )

    async def clear_sleep_stage_cache(self) -> None:
        if self._sleep_stage_cache is None:
            return
        try:
            await self._sleep_stage_cache.clear_all()
        except Exception as exc:
            logger.warning("清除睡眠阶段候选缓存失败：{}", exc)

    async def _recall_text_routes(
        self,
        *,
        request: SearchAudioRequest,
        query_vector: list[float],
        has_content_tags: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """并发执行相互独立的描述 KNN 与标签候选召回。"""
        description_recall = self._es_search.search_by_description_vector(
            query_vector,
            sleep_stage_tags=(
                request.sleep_stage_tags if self._settings.search_sleep_stage_filter_enabled else []
            ),
            size=_recall_size(request.top_k),
        )
        if not has_content_tags:
            return await description_recall, []

        desc_task = asyncio.create_task(description_recall)
        tag_task = asyncio.create_task(self._fetch_step1_candidates(request.sleep_stage_tags))
        try:
            desc_docs, tag_docs = await asyncio.gather(desc_task, tag_task)
            return desc_docs, tag_docs
        except BaseException:
            desc_task.cancel()
            tag_task.cancel()
            await asyncio.gather(desc_task, tag_task, return_exceptions=True)
            raise

    def _has_non_exact_content_candidate(
        self,
        candidates: list[dict[str, Any]],
        content_tags: list[str],
    ) -> bool:
        """存在精确标签未命中的候选时，才需要内容标签向量。"""
        if not candidates or not content_tags:
            return False
        requested = set(content_tags)
        return any(
            not self._es_search.parse_tags(doc).content_labels().intersection(requested)
            for doc in candidates
        )

    async def _prepare_content_admission_inputs(
        self,
        candidates: list[dict[str, Any]],
        request: SearchAudioRequest,
        *,
        need_dictionary: bool,
    ) -> tuple[
        DictionaryVectors,
        list[str],
        list[str],
        list[list[float]],
        list[list[float]],
    ]:
        """并行：词典向量预取 ∥ 厌恶归一 + 查询标签编码。"""
        dict_task = asyncio.create_task(
            self._prefetch_dictionary_vectors(candidates, required=need_dictionary)
        )
        vec_task = asyncio.create_task(self._encode_admission_query_vectors(request))
        try:
            dictionary_vectors, prepared = await asyncio.gather(dict_task, vec_task)
        except BaseException:
            dict_task.cancel()
            vec_task.cancel()
            await asyncio.gather(dict_task, vec_task, return_exceptions=True)
            raise
        content_tags, disliked_tags, content_vectors, dislike_vectors = prepared
        return dictionary_vectors, content_tags, disliked_tags, content_vectors, dislike_vectors

    async def _encode_admission_query_vectors(
        self,
        request: SearchAudioRequest,
    ) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
        """归一厌恶标签后编码内容/厌恶查询向量。"""
        tag_vectors = (
            await self._es_search.list_content_tag_vectors() if request.disliked_tags else []
        )
        disliked_tags = _usable_dislike_tags(
            _normalize_to_dictionary_labels(
                _normalize_color_noise_aliases(request.disliked_tags),
                tag_vectors,
            )
        )
        vector_disliked = _strip_voice_mention_tags(disliked_tags)
        content_tags = _normalize_color_noise_aliases(
            [tag for tag in request.content_tags if tag not in set(disliked_tags)]
        )
        content_vectors, dislike_vectors = await self._encode_request_tag_vectors(
            content_tags,
            vector_disliked,
        )
        return content_tags, disliked_tags, content_vectors, dislike_vectors

    async def _prefetch_dictionary_vectors(
        self,
        candidates: list[dict[str, Any]],
        *,
        required: bool,
    ) -> DictionaryVectors:
        """一次请求内去重预取候选标签向量，供内容与厌恶计算共同复用。"""
        if not required or not candidates:
            return {}

        candidate_keys: set[str] = set()
        unique_tag_ids: set[str] = set()
        for doc in candidates:
            candidate_keys.add(_candidate_key(doc))
            tags = self._es_search.parse_tags(doc)
            unique_tag_ids.update(EsSearch.content_tag_ids(tags))
        if not unique_tag_ids:
            return {}

        started_at = time.perf_counter()
        vectors = await self._es_search.get_dictionary_vectors(sorted(unique_tag_ids))
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        batch_size = max(1, self._settings.es_dictionary_mget_batch_size)
        batch_count = math.ceil(len(unique_tag_ids) / batch_size)
        logger.info(
            "标签向量请求级预取：候选数={}，唯一标签数={}，批次数={}，"
            "命中向量数={}，耗时={:.2f}毫秒",
            len(candidate_keys),
            len(unique_tag_ids),
            batch_count,
            len(vectors),
            elapsed_ms,
        )
        return vectors

    async def _encode_request_tag_vectors(
        self,
        content_tags: list[str],
        disliked_tags: list[str],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """一次 encode 同时产出内容标签与厌恶标签向量，减少 ONNX 调用次数。"""
        if not content_tags and not disliked_tags:
            return [], []
        unique_tags = _unique_preserve_order([*content_tags, *disliked_tags])
        encoded = await self._encode_texts(unique_tags)
        by_tag = dict(zip(unique_tags, encoded, strict=True))
        return (
            [by_tag[tag] for tag in content_tags],
            [by_tag[tag] for tag in disliked_tags],
        )

    async def _encode_one(self, text: str) -> list[float]:
        return (await self._encode_texts([text]))[0]

    async def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        """文本→向量，带进程内 LRU，命中则跳过 ONNX。"""
        if not texts:
            return []
        unique = _unique_preserve_order(texts)
        missing = [text for text in unique if text not in self._text_vector_cache]
        if missing:
            encoded = await self._encoder.encode(missing)
            for text, vector in zip(missing, encoded, strict=True):
                self._put_text_vector(text, vector)
        return [self._get_text_vector(text) for text in texts]

    def _put_text_vector(self, text: str, vector: list[float]) -> None:
        self._text_vector_cache[text] = vector
        self._text_vector_cache.move_to_end(text)
        while len(self._text_vector_cache) > self._text_vector_cache_max:
            self._text_vector_cache.popitem(last=False)

    def _get_text_vector(self, text: str) -> list[float]:
        vector = self._text_vector_cache[text]
        self._text_vector_cache.move_to_end(text)
        return vector

    async def _apply_content_admission(
        self,
        candidates: list[dict[str, Any]],
        content_tags: list[str],
        dictionary_vectors: DictionaryVectors,
        *,
        request_vectors: list[list[float]] | None = None,
    ) -> list[ScoredCandidate]:
        """步骤 2：无 content_tags 时跳过准入，保留睡眠阶段候选全集。"""
        if not content_tags:
            return [
                ScoredCandidate(
                    source=doc,
                    tags=self._es_search.parse_tags(doc),
                    match_count=0,
                    tag_score=0.0,
                    evidence_level=self._parse_evidence(doc),
                    recommend_weight=self._parse_weight(doc),
                )
                for doc in candidates
            ]

        request_vectors = (
            await self._encode_texts(content_tags) if request_vectors is None else request_vectors
        )
        admitted: list[ScoredCandidate] = []

        for doc in candidates:
            tags = self._es_search.parse_tags(doc)
            content_labels = tags.content_labels()

            exact_hits = content_labels.intersection(content_tags)
            if exact_hits:
                admitted.append(
                    ScoredCandidate(
                        source=doc,
                        tags=tags,
                        match_count=len(exact_hits),
                        evidence_level=self._parse_evidence(doc),
                        recommend_weight=self._parse_weight(doc),
                    )
                )
                continue

            vector_hits = self._count_fuzzy_vector_matches(
                tags,
                content_tags,
                request_vectors,
                dictionary_vectors,
            )
            if vector_hits > 0:
                admitted.append(
                    ScoredCandidate(
                        source=doc,
                        tags=tags,
                        match_count=vector_hits,
                        evidence_level=self._parse_evidence(doc),
                        recommend_weight=self._parse_weight(doc),
                    )
                )

        return admitted

    async def _score_content_candidates(
        self,
        candidates: list[dict[str, Any]],
        content_tags: list[str],
        dictionary_vectors: DictionaryVectors,
        *,
        request_vectors: list[list[float]] | None = None,
    ) -> list[ScoredCandidate]:
        """文本多路检索里的标签路：产出分数，不决定整体短路。"""
        if not candidates or not content_tags:
            return []

        request_vectors = (
            await self._encode_texts(content_tags) if request_vectors is None else request_vectors
        )
        scored: list[ScoredCandidate] = []
        tag_count = max(len(content_tags), 1)
        for doc in candidates:
            tags = self._es_search.parse_tags(doc)
            content_labels = tags.content_labels()
            exact_hits = content_labels.intersection(content_tags)
            vector_hits = 0
            if not exact_hits:
                vector_hits = self._count_fuzzy_vector_matches(
                    tags,
                    content_tags,
                    request_vectors,
                    dictionary_vectors,
                )
            match_count = len(exact_hits) if exact_hits else vector_hits
            if match_count <= 0:
                continue
            exact_score = len(exact_hits) / tag_count
            semantic_score = vector_hits / tag_count
            tag_score = min(1.0, 0.6 * exact_score + 0.4 * semantic_score)
            scored.append(
                ScoredCandidate(
                    source=doc,
                    tags=tags,
                    match_count=match_count,
                    tag_score=tag_score,
                    evidence_level=self._parse_evidence(doc),
                    recommend_weight=self._parse_weight(doc),
                )
            )
        return scored

    def _count_fuzzy_vector_matches(
        self,
        tags: AudioTags,
        request_tags: list[str],
        request_vectors: list[list[float]],
        dictionary_vectors: DictionaryVectors,
    ) -> int:
        """使用请求级向量快照计分；互斥标签只允许精确命中。"""
        tag_ids = EsSearch.content_tag_ids(tags)
        if not tag_ids or not request_vectors:
            return 0

        threshold = self._settings.sim_threshold
        matched = 0

        for request_tag, req_vec in zip(request_tags, request_vectors, strict=True):
            if request_tag in MUTUALLY_EXCLUSIVE_CONTENT_TAGS:
                continue
            for tid in tag_ids:
                doc_vec = dictionary_vectors.get(tid)
                if doc_vec and _cosine_similarity(req_vec, doc_vec) >= threshold:
                    matched += 1
                    break

        return matched

    async def _apply_dislike_filter(
        self,
        candidates: list[ScoredCandidate],
        disliked_tags: list[str],
        dictionary_vectors: DictionaryVectors,
        *,
        dislike_vectors: list[list[float]] | None = None,
    ) -> list[ScoredCandidate]:
        """步骤 3 前半：厌恶标签向量 vs 文档内容标签向量，余弦 ≥ SIM_THRESHOLD 则剔除。"""
        if not disliked_tags:
            return candidates

        dislike_vectors = (
            await self._encode_texts(disliked_tags)
            if dislike_vectors is None
            else dislike_vectors
        )
        result: list[ScoredCandidate] = []
        for candidate in candidates:
            if (
                self._count_fuzzy_vector_matches(
                    candidate.tags,
                    disliked_tags,
                    dislike_vectors,
                    dictionary_vectors,
                )
                > 0
            ):
                continue
            result.append(candidate)
        return result

    @staticmethod
    def _apply_coarse_rank(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        """步骤 3 后半：厌恶剔除后，按 match_count 写入粗排分 tag_score。"""
        for candidate in candidates:
            candidate.tag_score = float(candidate.match_count)
        return candidates

    async def _merge_and_rank_text_candidates(
        self,
        *,
        tag_candidates: list[ScoredCandidate],
        desc_candidates: list[ScoredCandidate],
        content_tags: list[str],
        disliked_tags: list[str],
        dictionary_vectors: DictionaryVectors,
        top_k: int | None,
        dislike_vectors: list[list[float]] | None = None,
        voice_filter_tags: list[str] | None = None,
    ) -> list[ScoredCandidate]:
        merged: dict[str, ScoredCandidate] = {}
        for candidate in [*tag_candidates, *desc_candidates]:
            key = _candidate_key(candidate.source)
            current = merged.get(key)
            if current is None:
                merged[key] = candidate
                continue
            current.tag_score = max(current.tag_score, candidate.tag_score)
            current.desc_score = max(current.desc_score, candidate.desc_score)
            current.match_count = max(current.match_count, candidate.match_count)

        dislike_vectors = (
            await self._encode_texts(disliked_tags)
            if disliked_tags and dislike_vectors is None
            else (dislike_vectors or [])
        )
        ranked: list[ScoredCandidate] = []
        for candidate in merged.values():
            penalty = self._dislike_penalty(
                candidate.tags,
                disliked_tags,
                dislike_vectors,
                dictionary_vectors,
            )
            if penalty >= 1.0:
                continue
            candidate.dislike_penalty = penalty
            candidate.final_score = self._final_score(
                candidate,
                has_content_tags=bool(content_tags),
            )
            ranked.append(candidate)

        ranked = _apply_voice_code_filter(
            ranked,
            disliked_tags=voice_filter_tags or disliked_tags,
        )
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    async def _extract_query_tags(
        self,
        query_text: str,
        query_vector: list[float],
        tag_vectors: list[dict[str, Any]],
    ) -> ExtractedQueryTags:
        if not tag_vectors:
            return ExtractedQueryTags(content_tags=[], disliked_tags=[])

        negative_fragments = _extract_negative_fragments(query_text)
        positive_text = _remove_negative_fragments(query_text)
        color_intents = _detect_color_noise_from_text(positive_text)
        exact_content_tags = _unique_preserve_order(
            [*color_intents, *_match_labels_from_text(positive_text, tag_vectors)]
        )
        content_tags = list(exact_content_tags)
        content_tags.extend(
            _similar_labels_from_vector(
                query_vector,
                tag_vectors,
                threshold=AUTO_TAG_SIM_THRESHOLD,
                exclude=set(content_tags),
                limit=AUTO_TAG_TOP_K - len(content_tags),
            )
        )
        content_tags = _remove_mutually_exclusive_expansions(
            content_tags,
            preferred_labels=exact_content_tags,
        )

        exact_disliked_tags = _match_labels_from_text(" ".join(negative_fragments), tag_vectors)
        disliked_tags = list(exact_disliked_tags)
        if negative_fragments:
            fragment_vectors = await self._encode_texts(negative_fragments)
            for vector in fragment_vectors:
                disliked_tags.extend(
                    _similar_labels_from_vector(
                        vector,
                        tag_vectors,
                        threshold=AUTO_DISLIKE_SIM_THRESHOLD,
                        exclude=set(disliked_tags),
                        limit=AUTO_TAG_TOP_K - len(disliked_tags),
                    )
                )
                if len(disliked_tags) >= AUTO_TAG_TOP_K:
                    break
        disliked_tags = _remove_mutually_exclusive_expansions(
            disliked_tags,
            preferred_labels=exact_disliked_tags,
        )

        return ExtractedQueryTags(
            content_tags=_unique_preserve_order(content_tags)[:AUTO_TAG_TOP_K],
            disliked_tags=_usable_dislike_tags(
                _unique_preserve_order(disliked_tags)[:AUTO_TAG_TOP_K]
            ),
        )

    def _dislike_penalty(
        self,
        tags: AudioTags,
        disliked_tags: list[str],
        dislike_vectors: list[list[float]],
        dictionary_vectors: DictionaryVectors,
    ) -> float:
        if not disliked_tags:
            return 0.0

        if tags.content_labels().intersection(disliked_tags):
            return 1.0

        max_similarity = self._max_fuzzy_vector_similarity(
            tags,
            disliked_tags,
            dislike_vectors,
            dictionary_vectors,
        )
        if max_similarity >= self._settings.strong_dislike_sim_threshold:
            return 1.0
        if max_similarity >= AUTO_DISLIKE_SIM_THRESHOLD:
            return WEAK_DISLIKE_PENALTY
        return 0.0

    @staticmethod
    def _max_fuzzy_vector_similarity(
        tags: AudioTags,
        request_tags: list[str],
        request_vectors: list[list[float]],
        dictionary_vectors: DictionaryVectors,
    ) -> float:
        tag_ids = EsSearch.content_tag_ids(tags)
        if not tag_ids or not request_vectors:
            return 0.0

        max_similarity = 0.0
        for request_tag, req_vec in zip(request_tags, request_vectors, strict=True):
            if request_tag in MUTUALLY_EXCLUSIVE_CONTENT_TAGS:
                continue
            for tag_id in tag_ids:
                doc_vec = dictionary_vectors.get(tag_id)
                if doc_vec:
                    max_similarity = max(max_similarity, _cosine_similarity(req_vec, doc_vec))
        return max_similarity

    def _candidate_from_doc(
        self,
        doc: dict[str, Any],
        *,
        desc_score: float = 0.0,
    ) -> ScoredCandidate:
        return ScoredCandidate(
            source=doc,
            tags=self._es_search.parse_tags(doc),
            desc_score=desc_score,
            evidence_level=self._parse_evidence(doc),
            recommend_weight=self._parse_weight(doc),
        )

    def _final_score(self, candidate: ScoredCandidate, *, has_content_tags: bool) -> float:
        evidence_score = EVIDENCE_WEIGHT_MAP.get(candidate.evidence_level, 0.45)
        recommend_score = _clamp01(candidate.recommend_weight)
        if has_content_tags:
            score = (
                0.35 * candidate.tag_score
                + 0.50 * candidate.desc_score
                + 0.10 * recommend_score
                + 0.05 * evidence_score
            )
        else:
            score = 0.75 * candidate.desc_score + 0.15 * recommend_score + 0.10 * evidence_score
        return score - candidate.dislike_penalty

    @staticmethod
    def _parse_evidence(doc: dict[str, Any]) -> EvidenceLevel:
        evidence_tags = doc.get("evidence_level_tags") or []
        if evidence_tags and isinstance(evidence_tags[0], dict):
            raw = str(evidence_tags[0].get("code", "C"))
        else:
            raw = str(doc.get("evidence_level", "C"))
        try:
            return EvidenceLevel(raw)
        except ValueError:
            return EvidenceLevel.C

    @staticmethod
    def _parse_weight(doc: dict[str, Any]) -> float:
        weight = doc.get("recommend_weight")
        if weight is not None:
            return float(weight)
        level = RetrievalService._parse_evidence(doc)
        return EVIDENCE_WEIGHT_MAP.get(level, 0.45)


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _log_tag_pipeline_timing(
    step1_ms: float,
    step2_ms: float,
    step3_ms: float,
    step4_ms: float,
    total_ms: float,
) -> None:
    """汇总标签检索四步耗时，并标出最慢步骤。"""
    steps = {
        "步骤1睡眠阶段": step1_ms,
        "步骤2内容准入": step2_ms,
        "步骤3厌恶粗排": step3_ms,
        "步骤4精排截断": step4_ms,
    }
    slowest = max(steps.items(), key=lambda item: item[1])[0]
    logger.info(
        "检索四步耗时：步骤1睡眠阶段={:.1f}毫秒，步骤2内容准入={:.1f}毫秒，"
        "步骤3厌恶粗排={:.1f}毫秒，步骤4精排截断={:.1f}毫秒，合计={:.1f}毫秒，最慢={}",
        step1_ms,
        step2_ms,
        step3_ms,
        step4_ms,
        total_ms,
        slowest,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """纯函数，便于单测；向量已 normalize 时等价于点积。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _candidate_key(source: dict[str, Any]) -> str:
    return str(
        source.get("_id") or source.get("id") or source.get("audio_url") or source.get("audio_name")
    )


def _tags_mention_voice(tags: list[str]) -> bool:
    return any(VOICE_MARKER in tag for tag in tags)


def _strip_voice_mention_tags(tags: list[str]) -> list[str]:
    return [tag for tag in tags if VOICE_MARKER not in tag]


def _usable_dislike_tags(tags: list[str]) -> list[str]:
    """去掉不应参与厌恶的标签（如全库共有的音工维名）。"""
    return [tag for tag in tags if tag not in IGNORED_DISLIKE_LABELS]


def _voice_value_code(source: dict[str, Any]) -> str | None:
    """读取音工「人声*」条目的 value.code；找不到返回 None。"""
    for item in source.get("audio_engineering_tags") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if VOICE_MARKER not in name:
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            return None
        code = value.get("code")
        if code is None:
            return None
        return str(code)
    return None


def _apply_voice_code_filter(
    candidates: list[ScoredCandidate],
    *,
    disliked_tags: list[str],
) -> list[ScoredCandidate]:
    """dislike 含人声时只留 value.code=none（无人声）；缺字段不过滤。"""
    if not _tags_mention_voice(disliked_tags):
        return candidates

    kept: list[ScoredCandidate] = []
    for candidate in candidates:
        code = _voice_value_code(candidate.source)
        if code is None or code == NO_VOICE_CODE:
            kept.append(candidate)
    return kept


def _parse_desc_score(doc: dict[str, Any]) -> float:
    return _clamp01(float(doc.get("_description_score") or 0.0))


def _recall_size(top_k: int | None) -> int:
    if top_k is None:
        return 100
    return min(max(top_k * 5, 50), 200)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _canonicalize_color_noise_label(label: str) -> str:
    """颜色噪音别名归一到规范标签；非颜色噪音标签原样返回。"""
    cleaned = label.strip()
    return COLOR_NOISE_ALIASES.get(cleaned.lower(), cleaned)


def _normalize_color_noise_aliases(labels: list[str]) -> list[str]:
    return _unique_preserve_order([_canonicalize_color_noise_label(label) for label in labels])


def _detect_color_noise_from_text(text: str) -> list[str]:
    """从自然语言中识别明确的颜色噪音诉求，返回规范标签。"""
    if not text:
        return []
    lowered = text.lower()
    found: list[str] = []
    for alias, canonical in COLOR_NOISE_ALIASES.items():
        if alias in lowered and canonical not in found:
            found.append(canonical)
    return found


def _remove_mutually_exclusive_expansions(
    labels: list[str],
    *,
    preferred_labels: list[str],
) -> list[str]:
    """指定具体颜色噪音后，丢弃自动扩出的父标签与兄弟标签。"""
    preferred = set(preferred_labels).intersection(MUTUALLY_EXCLUSIVE_CONTENT_TAGS)
    if not preferred:
        return labels
    blocked = (MUTUALLY_EXCLUSIVE_CONTENT_TAGS - preferred) | {COLOR_NOISE_PARENT_TAG}
    return [label for label in labels if label not in blocked]


def _has_mutually_exclusive_conflict(
    request_labels: list[str],
    candidate_labels: set[str],
) -> bool:
    requested = set(request_labels).intersection(MUTUALLY_EXCLUSIVE_CONTENT_TAGS)
    candidate = candidate_labels.intersection(MUTUALLY_EXCLUSIVE_CONTENT_TAGS)
    return bool(requested and candidate and requested.isdisjoint(candidate))


def _match_labels_from_text(text: str, tag_vectors: list[dict[str, Any]]) -> list[str]:
    """子串匹配词典标签：最短长度限制 + 优先更长标签，避免「低」「无」误伤。"""
    if not text:
        return []
    matched: list[tuple[int, str]] = []
    for item in tag_vectors:
        label = str(item["label"]).strip()
        if len(label) < MIN_AUTO_TAG_LABEL_LEN or label not in text:
            continue
        matched.append((text.index(label), label))
    matched.sort(key=lambda pair: (pair[0], -len(pair[1])))
    return _prefer_longer_labels([label for _, label in matched])


def _prefer_longer_labels(labels: list[str]) -> list[str]:
    """去掉已被更长标签覆盖的短标签（如已有「低动态」则丢弃「低」）。"""
    selected: list[str] = []
    for label in sorted(_unique_preserve_order(labels), key=len, reverse=True):
        if any(label != other and label in other for other in selected):
            continue
        selected.append(label)
    # 保持首次出现顺序
    order = {label: idx for idx, label in enumerate(_unique_preserve_order(labels))}
    selected.sort(key=lambda item: order.get(item, 0))
    return selected


def _normalize_to_dictionary_labels(
    raw_tags: list[str],
    tag_vectors: list[dict[str, Any]],
) -> list[str]:
    """自然语言厌恶句 → 词典标准标签；无法映射的长句不参与向量硬剔除。"""
    if not raw_tags:
        return []
    if not tag_vectors:
        return [
            tag.strip()
            for tag in raw_tags
            if MIN_AUTO_TAG_LABEL_LEN <= len(tag.strip()) <= MAX_FALLBACK_TAG_LEN
        ]

    dictionary = {
        str(item["label"]).strip()
        for item in tag_vectors
        if str(item.get("label", "")).strip()
    }
    normalized: list[str] = []
    for raw in raw_tags:
        tag = raw.strip()
        if not tag:
            continue
        if tag in dictionary:
            normalized.append(tag)
            continue
        cleaned = _remove_negative_fragments(tag)
        found = _match_labels_from_text(tag, tag_vectors)
        if not found and cleaned != tag:
            found = _match_labels_from_text(cleaned, tag_vectors)
        normalized.extend(found)
    return _unique_preserve_order(normalized)


def _similar_labels_from_vector(
    query_vector: list[float],
    tag_vectors: list[dict[str, Any]],
    *,
    threshold: float,
    exclude: set[str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    scored: list[tuple[float, str]] = []
    for item in tag_vectors:
        label = str(item["label"]).strip()
        vector = item.get("vector")
        if (
            not label
            or len(label) < MIN_AUTO_TAG_LABEL_LEN
            or label in exclude
            or not vector
        ):
            continue
        similarity = _cosine_similarity(query_vector, vector)
        if similarity >= threshold:
            scored.append((similarity, label))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return _prefer_longer_labels([label for _, label in scored])[:limit]


def _extract_negative_fragments(query_text: str) -> list[str]:
    fragments: list[str] = []
    for marker in NEGATIVE_MARKERS:
        pattern = rf"{re.escape(marker)}([^，,。.;；!！?？]{{1,12}})"
        for match in re.finditer(pattern, query_text):
            fragment = match.group(1).strip()
            if fragment:
                fragments.append(fragment)
    return _unique_preserve_order(fragments)


def _remove_negative_fragments(query_text: str) -> str:
    cleaned = query_text
    for marker in NEGATIVE_MARKERS:
        pattern = rf"{re.escape(marker)}[^，,。.;；!！?？]{{0,12}}"
        cleaned = re.sub(pattern, " ", cleaned)
    return cleaned
