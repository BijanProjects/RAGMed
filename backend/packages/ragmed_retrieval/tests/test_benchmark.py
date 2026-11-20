from collections.abc import Sequence
from itertools import count
from pathlib import Path

import numpy as np

from ragmed_retrieval.benchmark import (
    BenchmarkResult,
    Pipeline,
    corpus_fingerprint,
    result_filename,
    results_table,
    run_benchmark,
    top_k,
)
from ragmed_retrieval.chunking import Chunk
from ragmed_retrieval.embedding import l2_normalize
from ragmed_retrieval.evalset import EvalQuery
from ragmed_retrieval.hybrid import BM25Retriever
from ragmed_retrieval.metrics import RetrievalScores
from ragmed_retrieval.testing import HashingEmbedder

CHUNKS = [
    Chunk("p1:0", "p1", 0, "symptoms", "q", "asthma causes wheezing and shortness of breath"),
    Chunk("p1:1", "p1", 1, "symptoms", "q", "asthma attacks may need an inhaler"),
    Chunk("p2:0", "p2", 0, "treatment", "q", "diabetes is managed with insulin and diet"),
    Chunk("p3:0", "p3", 0, "causes", "q", "influenza is caused by a virus"),
]
QUERIES = [
    EvalQuery("e1", "what helps asthma wheezing", "symptoms", ("p1",)),
    EvalQuery("e2", "insulin for diabetes", "treatment", ("p2",)),
    EvalQuery("e3", "virus behind influenza", "causes", ("p3",)),
]


def test_top_k_orders_by_similarity() -> None:
    docs = l2_normalize(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    query = l2_normalize(np.array([[1, 0.1]], dtype=np.float32))

    assert top_k(query, docs, k=3).tolist() == [[0, 2, 1]]
    assert top_k(query, docs, k=10).shape == (1, 3)


def test_benchmark_scores_obvious_matches_perfectly() -> None:
    ticks = count(step=0.002)

    result = run_benchmark(HashingEmbedder(), CHUNKS, QUERIES, clock=lambda: next(ticks))

    assert result.model == "test/hashing"
    assert result.chunks == 4
    assert result.dimensions == 64
    assert result.scores.hit_at_1 == 1.0
    assert result.scores.mrr == 1.0
    assert result.query_p95_ms == 2.0


def test_document_vectors_are_cached(tmp_path: Path) -> None:
    embedder = HashingEmbedder()

    first = run_benchmark(embedder, CHUNKS, QUERIES, cache_dir=tmp_path)
    second = run_benchmark(embedder, CHUNKS, QUERIES, cache_dir=tmp_path)

    assert embedder.document_calls == 1
    assert first.scores == second.scores
    assert second.index_seconds == 0.0


def test_fingerprint_changes_with_content() -> None:
    changed = [*CHUNKS[:-1], Chunk("p3:0", "p3", 0, "causes", "q", "different text")]

    assert corpus_fingerprint(CHUNKS) != corpus_fingerprint(changed)


def _result(model: str, ndcg: float) -> BenchmarkResult:
    scores = RetrievalScores(10, 0.5, 0.6, 0.7, ndcg, 0.55)
    return BenchmarkResult(model, 100, 384, 1.0, 5.0, 9.0, scores)


def test_results_table_sorts_by_ndcg() -> None:
    table = results_table([_result("small", 0.4), _result("large", 0.8)])

    lines = table.strip().splitlines()
    assert lines[0].startswith("| Model |")
    assert "`large`" in lines[2]
    assert "`small`" in lines[3]
    assert "0.800" in lines[2]


def test_result_serialises_to_json() -> None:
    assert '"ndcg_at_10": 0.8' in _result("m", 0.8).to_json()


class ReverseReranker:
    """Puts candidates in reverse id order, which a perfect ranking cannot survive."""

    name = "org/reverse-reranker"

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        return [float(i) for i in range(len(passages))]


def test_hybrid_pipeline_is_labelled_and_scored() -> None:
    pipeline = Pipeline(sparse=BM25Retriever([c.text for c in CHUNKS]))

    result = run_benchmark(HashingEmbedder(), CHUNKS, QUERIES, pipeline=pipeline)

    assert result.pipeline == "hybrid"
    assert result.reranker is None
    assert result.scores.hit_at_1 == 1.0
    assert result_filename(result) == "test__hashing--hybrid.json"


def test_reranker_changes_the_ranking() -> None:
    pipeline = Pipeline(reranker=ReverseReranker(), rerank_depth=4)

    result = run_benchmark(HashingEmbedder(), CHUNKS, QUERIES, pipeline=pipeline)

    assert result.pipeline == "dense+rerank"
    assert result.reranker == "org/reverse-reranker"
    assert result.scores.hit_at_1 < 1.0
    assert result_filename(result) == "test__hashing--dense+rerank--reverse-reranker.json"
    assert "dense+rerank (`reverse-reranker`)" in results_table([result])
