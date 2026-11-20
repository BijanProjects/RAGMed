"""Binary-relevance ranking metrics for retrieval evaluation."""

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean


def _check(relevant: Collection[str], k: int | None = None) -> None:
    if not relevant:
        raise ValueError("relevant must not be empty")
    if k is not None and k <= 0:
        raise ValueError("k must be positive")


def dedupe(ranked: Iterable[str]) -> list[str]:
    """Collapse repeated ids (e.g. several chunks of one answer), keeping first rank."""
    return list(dict.fromkeys(ranked))


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(relevant, k)
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(relevant, k)
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    _check(relevant)
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(relevant, k)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    queries: int
    hit_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    mrr: float


def score_run(results: Sequence[tuple[Sequence[str], Collection[str]]]) -> RetrievalScores:
    """Average metrics over (ranked ids, relevant ids) pairs, one per query."""
    if not results:
        raise ValueError("results must not be empty")
    return RetrievalScores(
        queries=len(results),
        hit_at_1=fmean(hit_at_k(r, rel, 1) for r, rel in results),
        recall_at_5=fmean(recall_at_k(r, rel, 5) for r, rel in results),
        recall_at_10=fmean(recall_at_k(r, rel, 10) for r, rel in results),
        ndcg_at_10=fmean(ndcg_at_k(r, rel, 10) for r, rel in results),
        mrr=fmean(reciprocal_rank(r, rel) for r, rel in results),
    )
