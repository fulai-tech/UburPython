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
