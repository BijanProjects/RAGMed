import math

import pytest

from ragmed_retrieval.metrics import (
    dedupe,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_run,
)

RANKED = ["d3", "d1", "d7", "d2", "d9"]


def test_dedupe_keeps_first_occurrence() -> None:
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, {"d1", "d2"}, k=2) == 0.5
    assert recall_at_k(RANKED, {"d1", "d2"}, k=5) == 1.0
    assert recall_at_k(RANKED, {"d8"}, k=5) == 0.0


def test_hit_at_k() -> None:
    assert hit_at_k(RANKED, {"d3"}, k=1) == 1.0
    assert hit_at_k(RANKED, {"d1"}, k=1) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, {"d7", "d2"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"d8"}) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    # Relevant at ranks 2 and 4; ideal puts them at ranks 1 and 2.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)

    assert ndcg_at_k(RANKED, {"d1", "d2"}, k=5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["d1", "d2"], {"d1", "d2"}, k=10) == pytest.approx(1.0)


@pytest.mark.parametrize("func", [recall_at_k, hit_at_k, ndcg_at_k])
def test_invalid_inputs(func: object) -> None:
    with pytest.raises(ValueError, match="relevant"):
        func(RANKED, set(), 5)  # type: ignore[operator]
    with pytest.raises(ValueError, match="k must be positive"):
        func(RANKED, {"d1"}, 0)  # type: ignore[operator]


def test_score_run_averages_over_queries() -> None:
    scores = score_run([(["a", "b"], {"a"}), (["b", "a"], {"a"})])

    assert scores.queries == 2
    assert scores.hit_at_1 == 0.5
    assert scores.recall_at_5 == 1.0
    assert scores.mrr == pytest.approx(0.75)


def test_score_run_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        score_run([])
