"""Shared stateless hybrid recall; callers retain their admission and ranking policies."""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.codes import HttpStatus
from app.core.exceptions import AppError
from app.server.somni.audio.rrf import rrf_merge

RECALL_VERSION = "prescription_hybrid.v1"


async def encode_query(encoder: Any, text: str, language: str) -> list[float]:
    """Keep model selection and vector field selection on the same language basis."""
    return await encoder.encode_one(text, language=language)


async def gather_lanes(
    jobs: list[tuple[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """One failed lane does not discard the other; caller cancellation propagates."""
    results = await asyncio.gather(*(job for _, job in jobs), return_exceptions=True)
    lanes: dict[str, list[dict[str, Any]]] = {"bm25": [], "vector": []}
    failed = []
    for (name, _), result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException):
            failed.append(name)
        else:
            lanes[name] = result
    return lanes["bm25"], lanes["vector"], failed


def require_available_lane(failed: list[str]) -> None:
    if {"bm25", "vector"} <= set(failed):
        raise AppError(message="音频混合检索暂不可用", status_code=HttpStatus.SERVICE_UNAVAILABLE)


def merge_documents(
    bm25: list[dict[str, Any]], vector: list[dict[str, Any]], *, k: int = 60
) -> list[dict[str, Any]]:
    """RRF orders recall only. Vector similarity is never synthesized from BM25."""
    by_id: dict[str, dict[str, Any]] = {}
    lists = []
    for docs in (bm25, vector):
        ids = []
        for doc in docs:
            identifier = str(doc.get("_id") or doc.get("id") or "")
            if not identifier:
                continue
            if identifier not in ids:
                ids.append(identifier)
            existing = by_id.setdefault(identifier, {})
            for field, value in doc.items():
                if field not in existing or field in {
                    "_description_score",
                    "_knn_score",
                    "_bm25_score",
                }:
                    existing[field] = value
        lists.append(ids)
    return [
        {**by_id[identifier], "_rrf_score": score, "_retrieval_rank": rank}
        for rank, (identifier, score) in enumerate(rrf_merge(lists, k=k))
    ]
