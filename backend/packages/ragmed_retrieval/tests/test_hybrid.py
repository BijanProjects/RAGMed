from collections.abc import Sequence

import pytest

from ragmed_retrieval.hybrid import BM25Retriever, rerank_head, rrf_fuse

DOCS = [
    "Asthma causes wheezing and shortness of breath.",
    "Diabetes is treated with insulin and a careful diet.",
    "Influenza is caused by a virus that spreads in winter.",
]


class LengthReranker:
    """Scores shorter passages higher, so the expected order is easy to see."""

    name = "test/length"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        self.calls.append(list(passages))
        return [-len(p) for p in passages]


def test_rrf_rewards_documents_ranked_well_by_both_lists() -> None:
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]])

    assert fused[0] == 1
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_breaks_ties_by_id() -> None:
    assert rrf_fuse([[5], [2]]) == [2, 5]


def test_rerank_head_reorders_only_the_head() -> None:
    texts = ["long passage", "mid one", "s", "tail"]
    reranker = LengthReranker()

    order = rerank_head("q", [0, 1, 2, 3], texts, reranker, depth=3)

    assert order == [2, 1, 0, 3]
    assert reranker.calls == [["long passage", "mid one", "s"]]


def test_rerank_head_with_no_candidates() -> None:
    assert rerank_head("q", [], [], LengthReranker(), depth=5) == []


def test_bm25_matches_stemmed_terms() -> None:
    bm25 = BM25Retriever(DOCS)

    results = bm25.search(["treating diabetics with insulin", "viruses"], k=10)

    assert results[0][0] == 1
    assert results[1] == [2]


def test_bm25_returns_nothing_for_unknown_words() -> None:
    assert BM25Retriever(DOCS).search(["zzzz"], k=2) == [[]]


@pytest.mark.parametrize("k", [1, 50])
def test_bm25_caps_k_at_corpus_size(k: int) -> None:
    assert len(BM25Retriever(DOCS).search(["asthma diabetes influenza"], k=k)[0]) == min(k, 3)
