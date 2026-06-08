"""三维度检索流水线（RetrievalService）。

顺序固定（规范 §五，不可调换）：
  1. 睡眠阶段精确过滤（ES term，无命中 → 空数组）
  2. 内容形态准入（精确交集 或 向量逐标签计分，余弦 ≥ SIM_THRESHOLD）
  3. 厌恶剔除 + 粗排（向量余弦 ≥ SIM_THRESHOLD 则剔除）
  4. 精排（当前等效 match_count；evidence_level 权重待业务完善）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.core.config import Settings
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.schemas.audio import (
    AudioResult,
    AudioTags,
    EvidenceLevel,
    EVIDENCE_WEIGHT_MAP,
    SearchAudioRequest,
)


@dataclass
class ScoredCandidate:
    """流水线中间态：ES 文档 + 解析后的标签 + 粗排分数。"""

    source: dict[str, Any]
    tags: AudioTags
    match_count: int = 0
    evidence_level: EvidenceLevel = EvidenceLevel.C
    recommend_weight: float = 0.45


class RetrievalService:
    """三维度检索：睡眠阶段 → 内容形态 → 厌恶剔除 → 精排。"""

    def __init__(
        self,
        es_search: EsSearch,
        encoder: Encoder,
        settings: Settings,
    ) -> None:
        self._es_search = es_search
        self._encoder = encoder
        self._settings = settings

    async def search(self, request: SearchAudioRequest) -> list[AudioResult]:
        top_k_label = request.top_k if request.top_k is not None else "全部"
        logger.info(
            "检索开始，睡眠阶段={}，内容标签={}，厌恶标签={}，top_k={}",
            request.sleep_stage_tags,
            request.content_tags,
            request.disliked_tags,
            top_k_label,
        )

        candidates_raw = await self._fetch_step1_candidates(request.sleep_stage_tags)
        if not candidates_raw:
            if self._settings.search_sleep_stage_filter_enabled:
                logger.info("检索：睡眠阶段无匹配，短路返回空结果")
            return []

        admitted = await self._apply_content_admission(candidates_raw, request.content_tags)
        logger.info("检索步骤2/4 内容形态准入：通过数={}", len(admitted))
        if not admitted:
            logger.info("检索：内容形态准入无匹配，短路返回空结果")
            return []

        filtered = await self._apply_dislike_and_coarse_rank(admitted, request.disliked_tags)
        logger.info("检索步骤3/4 厌恶剔除+粗排：剩余数={}", len(filtered))

        # 精排：业务字段未完善前，sort key 等价于 match_count
        ranked = sorted(filtered, key=lambda c: c.match_count, reverse=True)
        if request.top_k is not None:
            ranked = ranked[: request.top_k]
        logger.info(
            "检索步骤4/4 精排截断：top_k={}，输出数={}，match_count序列={}",
            top_k_label,
            len(ranked),
            [c.match_count for c in ranked],
        )

        results = [
            AudioResult(
                audio_url=c.source["audio_url"],
                audio_name=c.source["audio_name"],
                tags=c.tags.to_label_tags(),
                evidence_level=c.evidence_level,
                recommend_weight=c.recommend_weight,
            )
            for c in ranked
        ]
        logger.info("检索完成，命中数={}", len(results))
        return results

    async def _fetch_step1_candidates(self, sleep_stage_tags: list[str]) -> list[dict[str, Any]]:
        """步骤 1：按配置决定是否按睡眠阶段过滤。"""
        if not self._settings.search_sleep_stage_filter_enabled:
            candidates = await self._es_search.list_all_audio_candidates()
            logger.info("检索步骤1/4 睡眠阶段过滤：已跳过，候选数={}", len(candidates))
            return candidates

        candidates = await self._es_search.filter_by_sleep_stage(sleep_stage_tags)
        logger.info("检索步骤1/4 睡眠阶段过滤：候选数={}", len(candidates))
        return candidates

    async def _apply_content_admission(
        self,
        candidates: list[dict[str, Any]],
        content_tags: list[str],
    ) -> list[ScoredCandidate]:
        """步骤 2：无 content_tags 时跳过准入，保留睡眠阶段候选全集。"""
        if not content_tags:
            return [
                ScoredCandidate(
                    source=doc,
                    tags=self._es_search.parse_tags(doc),
                    match_count=0,
                    evidence_level=self._parse_evidence(doc),
                    recommend_weight=self._parse_weight(doc),
                )
                for doc in candidates
            ]

        request_vectors = await self._encoder.encode(content_tags)
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

            # 精确未命中才走向量模糊；同义词优先归一，不单独上向量（规范 §五）
            vector_hits = await self._count_fuzzy_vector_matches(tags, request_vectors)
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

    async def _count_fuzzy_vector_matches(
        self,
        tags: AudioTags,
        request_vectors: list[list[float]],
    ) -> int:
        """每个请求标签向量独立计分：与文档四维度 tag_vectors 任一 ≥ SIM_THRESHOLD 则 +1。"""
        vector_ids: list[str] = []
        for dim in (tags.content_form, tags.mechanism, tags.audio_feat, tags.rhythm):
            vector_ids.extend(item.vector_id for item in dim)

        if not vector_ids or not request_vectors:
            return 0

        stored = await self._es_search.get_tag_vectors(vector_ids)
        threshold = self._settings.sim_threshold
        matched = 0

        for req_vec in request_vectors:
            for vid in vector_ids:
                doc_vec = stored.get(vid)
                if doc_vec and _cosine_similarity(req_vec, doc_vec) >= threshold:
                    matched += 1
                    break

        return matched

    async def _apply_dislike_and_coarse_rank(
        self,
        candidates: list[ScoredCandidate],
        disliked_tags: list[str],
    ) -> list[ScoredCandidate]:
        """步骤 3：厌恶标签向量 vs 文档内容标签向量，余弦 ≥ SIM_THRESHOLD 则剔除。"""
        if not disliked_tags:
            return candidates

        dislike_vectors = await self._encoder.encode(disliked_tags)
        result: list[ScoredCandidate] = []
        for candidate in candidates:
            if await self._count_fuzzy_vector_matches(candidate.tags, dislike_vectors) > 0:
                continue
            result.append(candidate)
        return result

    @staticmethod
    def _parse_evidence(doc: dict[str, Any]) -> EvidenceLevel:
        raw = doc.get("evidence_level", "C")
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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """纯函数，便于单测；向量已 normalize 时等价于点积。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
