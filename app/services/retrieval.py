"""三维度检索流水线（RetrievalService）。

顺序固定（规范 §五，不可调换）：
  1. 睡眠阶段精确过滤（ES term，无命中 → 空数组）
  2. 内容形态准入（精确交集 或 向量逐标签计分，余弦 ≥ SIM_THRESHOLD）
  3. 厌恶剔除 + 粗排（向量余弦 ≥ SIM_THRESHOLD 则剔除）
  4. 精排（当前等效 match_count；evidence_level 权重待业务完善）
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

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

AUTO_TAG_TOP_K = 5
AUTO_TAG_SIM_THRESHOLD = 0.62
AUTO_DISLIKE_SIM_THRESHOLD = 0.65
STRONG_DISLIKE_SIM_THRESHOLD = 0.78
WEAK_DISLIKE_PENALTY = 0.2
NEGATIVE_MARKERS = ("不要", "避免", "讨厌", "不喜欢", "别")


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

    async def search(self, request: SearchAudioRequest) -> list[dict[str, Any]]:
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

        ranked = sorted(filtered, key=lambda c: c.tag_score, reverse=True)
        if request.top_k is not None:
            ranked = ranked[: request.top_k]
        logger.info(
            "检索步骤4/4 精排截断：top_k={}，输出数={}，match_count序列={}",
            top_k_label,
            len(ranked),
            [c.match_count for c in ranked],
        )

        results = [c.source for c in ranked]
        logger.info("检索完成，命中数={}", len(results))
        return results

    async def _search_text_multi_route(
        self,
        request: SearchAudioRequest,
        query_text: str,
    ) -> list[dict[str, Any]]:
        top_k_label = request.top_k if request.top_k is not None else "全部"
        query_vector = await self._encoder.encode_one(query_text)
        extracted = await self._extract_query_tags(query_text, query_vector)
        content_tags = _unique_preserve_order([*request.content_tags, *extracted.content_tags])
        disliked_tags = _unique_preserve_order([*request.disliked_tags, *extracted.disliked_tags])
        logger.info(
            "多路检索开始，query_text={}，显式内容标签={}，自动内容标签={}，"
            "显式厌恶标签={}，自动厌恶标签={}，top_k={}",
            query_text,
            request.content_tags,
            extracted.content_tags,
            request.disliked_tags,
            extracted.disliked_tags,
            top_k_label,
        )

        recall_size = _recall_size(request.top_k)
        desc_docs = await self._es_search.search_by_description_vector(
            query_vector,
            sleep_stage_tags=(
                request.sleep_stage_tags if self._settings.search_sleep_stage_filter_enabled else []
            ),
            size=recall_size,
        )
        desc_candidates = [
            self._candidate_from_doc(doc, desc_score=_parse_desc_score(doc)) for doc in desc_docs
        ]

        tag_candidates: list[ScoredCandidate] = []
        if content_tags:
            tag_docs = await self._fetch_step1_candidates(request.sleep_stage_tags)
            tag_candidates = await self._score_content_candidates(tag_docs, content_tags)

        merged = await self._merge_and_rank_text_candidates(
            tag_candidates=tag_candidates,
            desc_candidates=desc_candidates,
            content_tags=content_tags,
            disliked_tags=disliked_tags,
            top_k=request.top_k,
        )
        logger.info(
            "多路检索完成，标签召回={}，描述召回={}，融合输出={}，final_score序列={}",
            len(tag_candidates),
            len(desc_candidates),
            len(merged),
            [round(c.final_score, 4) for c in merged],
        )
        return [c.source for c in merged]

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
                    tag_score=0.0,
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
                        tag_score=float(len(exact_hits)),
                        evidence_level=self._parse_evidence(doc),
                        recommend_weight=self._parse_weight(doc),
                    )
                )
                continue

            vector_hits = await self._count_fuzzy_vector_matches(tags, request_vectors)
            if vector_hits > 0:
                admitted.append(
                    ScoredCandidate(
                        source=doc,
                        tags=tags,
                        match_count=vector_hits,
                        tag_score=float(vector_hits),
                        evidence_level=self._parse_evidence(doc),
                        recommend_weight=self._parse_weight(doc),
                    )
                )

        return admitted

    async def _score_content_candidates(
        self,
        candidates: list[dict[str, Any]],
        content_tags: list[str],
    ) -> list[ScoredCandidate]:
        """文本多路检索里的标签路：产出分数，不决定整体短路。"""
        if not candidates or not content_tags:
            return []

        request_vectors = await self._encoder.encode(content_tags)
        scored: list[ScoredCandidate] = []
        tag_count = max(len(content_tags), 1)
        for doc in candidates:
            tags = self._es_search.parse_tags(doc)
            content_labels = tags.content_labels()
            exact_hits = content_labels.intersection(content_tags)
            vector_hits = 0
            if not exact_hits:
                vector_hits = await self._count_fuzzy_vector_matches(tags, request_vectors)
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

    async def _count_fuzzy_vector_matches(
        self,
        tags: AudioTags,
        request_vectors: list[list[float]],
    ) -> int:
        """每个请求标签向量独立计分：与文档 tag_dictionary name_vector ≥ SIM_THRESHOLD 则 +1。"""
        tag_ids = EsSearch.content_tag_ids(tags)
        if not tag_ids or not request_vectors:
            return 0

        stored = await self._es_search.get_dictionary_vectors(tag_ids)
        threshold = self._settings.sim_threshold
        matched = 0

        for req_vec in request_vectors:
            for tid in tag_ids:
                doc_vec = stored.get(tid)
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

    async def _merge_and_rank_text_candidates(
        self,
        *,
        tag_candidates: list[ScoredCandidate],
        desc_candidates: list[ScoredCandidate],
        content_tags: list[str],
        disliked_tags: list[str],
        top_k: int | None,
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

        dislike_vectors = await self._encoder.encode(disliked_tags) if disliked_tags else []
        ranked: list[ScoredCandidate] = []
        for candidate in merged.values():
            penalty = await self._dislike_penalty(candidate.tags, disliked_tags, dislike_vectors)
            if penalty >= 1.0:
                continue
            candidate.dislike_penalty = penalty
            candidate.final_score = self._final_score(
                candidate,
                has_content_tags=bool(content_tags),
            )
            ranked.append(candidate)

        ranked.sort(key=lambda c: c.final_score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    async def _extract_query_tags(
        self,
        query_text: str,
        query_vector: list[float],
    ) -> ExtractedQueryTags:
        tag_vectors = await self._es_search.list_content_tag_vectors()
        if not tag_vectors:
            return ExtractedQueryTags(content_tags=[], disliked_tags=[])

        negative_fragments = _extract_negative_fragments(query_text)
        positive_text = _remove_negative_fragments(query_text)
        content_tags = _match_labels_from_text(positive_text, tag_vectors)
        content_tags.extend(
            _similar_labels_from_vector(
                query_vector,
                tag_vectors,
                threshold=AUTO_TAG_SIM_THRESHOLD,
                exclude=set(content_tags),
                limit=AUTO_TAG_TOP_K - len(content_tags),
            )
        )

        disliked_tags = _match_labels_from_text(" ".join(negative_fragments), tag_vectors)
        if negative_fragments:
            fragment_vectors = await self._encoder.encode(negative_fragments)
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

        return ExtractedQueryTags(
            content_tags=_unique_preserve_order(content_tags)[:AUTO_TAG_TOP_K],
            disliked_tags=_unique_preserve_order(disliked_tags)[:AUTO_TAG_TOP_K],
        )

    async def _dislike_penalty(
        self,
        tags: AudioTags,
        disliked_tags: list[str],
        dislike_vectors: list[list[float]],
    ) -> float:
        if not disliked_tags:
            return 0.0

        if tags.content_labels().intersection(disliked_tags):
            return 1.0

        max_similarity = await self._max_fuzzy_vector_similarity(tags, dislike_vectors)
        if max_similarity >= STRONG_DISLIKE_SIM_THRESHOLD:
            return 1.0
        if max_similarity >= AUTO_DISLIKE_SIM_THRESHOLD:
            return WEAK_DISLIKE_PENALTY
        return 0.0

    async def _max_fuzzy_vector_similarity(
        self,
        tags: AudioTags,
        request_vectors: list[list[float]],
    ) -> float:
        tag_ids = EsSearch.content_tag_ids(tags)
        if not tag_ids or not request_vectors:
            return 0.0

        stored = await self._es_search.get_dictionary_vectors(tag_ids)
        max_similarity = 0.0
        for req_vec in request_vectors:
            for tag_id in tag_ids:
                doc_vec = stored.get(tag_id)
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
            score = (
                0.75 * candidate.desc_score
                + 0.15 * recommend_score
                + 0.10 * evidence_score
            )
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
        source.get("_id")
        or source.get("id")
        or source.get("audio_url")
        or source.get("audio_name")
    )


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


def _match_labels_from_text(text: str, tag_vectors: list[dict[str, Any]]) -> list[str]:
    matched: list[tuple[int, str]] = []
    for item in tag_vectors:
        label = str(item["label"]).strip()
        if label and label in text:
            matched.append((text.index(label), label))
    matched.sort(key=lambda pair: pair[0])
    return _unique_preserve_order([label for _, label in matched])


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
        if not label or label in exclude or not vector:
            continue
        similarity = _cosine_similarity(query_vector, vector)
        if similarity >= threshold:
            scored.append((similarity, label))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return _unique_preserve_order([label for _, label in scored])[:limit]


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
