import pytest

from app.server.somni.audio.rrf import rrf_merge


def test_rrf_merge_sums_ranks_and_sorts() -> None:
    # A: rank1 in list0, rank2 in list1 → high
    # B: only list0 rank2
    # C: only list1 rank1
    merged = rrf_merge([["A", "B"], ["C", "A"]], k=60)
    ids = [doc_id for doc_id, _ in merged]
    assert ids[0] == "A"
    assert set(ids) == {"A", "B", "C"}
    scores = dict(merged)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["C"] == pytest.approx(1 / 61)
    assert scores["B"] == pytest.approx(1 / 62)


def test_rrf_merge_tie_breaks_by_doc_id() -> None:
    # 两路各只有一个不同 id，同分 → id 升序
    merged = rrf_merge([["b"], ["a"]], k=60)
    assert [doc_id for doc_id, _ in merged] == ["a", "b"]


def test_rrf_merge_empty() -> None:
    assert rrf_merge([]) == []
    assert rrf_merge([[], []]) == []


def test_filter_docs_in_both_lanes_keeps_intersection_only() -> None:
    from app.server.somni.audio.catalog import _filter_docs_in_both_lanes

    docs = [
        {"_id": "both", "_bm25_score": 2.0, "_knn_score": 0.9, "_rrf_score": 0.03},
        {"_id": "bm25_only", "_bm25_score": 1.5, "_rrf_score": 0.016},
        {"_id": "knn_only", "_knn_score": 0.88, "_rrf_score": 0.016},
    ]
    kept = _filter_docs_in_both_lanes(docs)
    assert [d["_id"] for d in kept] == ["both"]


def test_rerank_by_relevance_puts_strong_knn_first() -> None:
    from app.server.somni.audio.catalog import _rerank_hybrid_docs_by_relevance

    docs = [
        {
            "_id": "dual_mid",
            "audio_name": "热带细雨",
            "_bm25_score": 2.5,
            "_knn_score": 0.836,
            "_rrf_score": 0.031,
        },
        {"_id": "knn_top", "audio_name": "雨幕绵延", "_knn_score": 0.849, "_rrf_score": 0.016},
        {"_id": "bm25_only", "audio_name": "回想", "_bm25_score": 2.3, "_rrf_score": 0.016},
    ]
    ranked = _rerank_hybrid_docs_by_relevance(docs)
    assert [d["audio_name"] for d in ranked] == ["雨幕绵延", "热带细雨", "回想"]
