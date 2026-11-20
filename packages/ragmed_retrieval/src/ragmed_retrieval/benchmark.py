"""Dense-retrieval benchmark: embed the corpus once, then score the evaluation set."""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ragmed_retrieval.chunking import Chunk
from ragmed_retrieval.embedding import Embedder, Vectors
from ragmed_retrieval.evalset import EvalQuery
from ragmed_retrieval.metrics import RetrievalScores, dedupe, score_run

CANDIDATE_POOL = 100
LATENCY_SAMPLE = 100


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model: str
    chunks: int
    dimensions: int
    index_seconds: float
    query_p50_ms: float
    query_p95_ms: float
    scores: RetrievalScores

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()[:12]


def top_k(query_vectors: Vectors, doc_vectors: Vectors, k: int) -> np.ndarray:
    """Indices of the k most similar documents per query (vectors are normalised)."""
    k = min(k, doc_vectors.shape[0])
    sims = query_vectors @ doc_vectors.T
    candidates = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sims, candidates, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(candidates, order, axis=1)


def _cached_document_vectors(
    embedder: Embedder, chunks: Sequence[Chunk], cache_dir: Path | None
) -> tuple[Vectors, float]:
    path = None
    if cache_dir is not None:
        slug = embedder.name.replace("/", "__")
        path = cache_dir / f"{slug}-{corpus_fingerprint(chunks)}.npy"
        if path.exists():
            return np.load(path), 0.0

    start = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    elapsed = time.perf_counter() - start
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    return vectors, elapsed


def _query_latency_ms(
    embedder: Embedder, queries: Sequence[EvalQuery], clock: Callable[[], float]
) -> tuple[float, float]:
    timings = []
    for query in queries[:LATENCY_SAMPLE]:
        start = clock()
        embedder.embed_queries([query.query])
        timings.append((clock() - start) * 1000)
    p50, p95 = np.percentile(timings, [50, 95])
    return float(p50), float(p95)


def run_benchmark(
    embedder: Embedder,
    chunks: Sequence[Chunk],
    queries: Sequence[EvalQuery],
    *,
    cache_dir: Path | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    doc_vectors, index_seconds = _cached_document_vectors(embedder, chunks, cache_dir)
    query_vectors = embedder.embed_queries([q.query for q in queries])

    ranked_indices = top_k(query_vectors, doc_vectors, CANDIDATE_POOL)
    results = [
        (dedupe(chunks[i].parent_id for i in row), query.relevant_ids)
        for row, query in zip(ranked_indices, queries, strict=True)
    ]
    p50, p95 = _query_latency_ms(embedder, queries, clock)

    return BenchmarkResult(
        model=embedder.name,
        chunks=len(chunks),
        dimensions=int(doc_vectors.shape[1]),
        index_seconds=round(index_seconds, 1),
        query_p50_ms=round(p50, 1),
        query_p95_ms=round(p95, 1),
        scores=score_run(results),
    )


def results_table(results: Sequence[BenchmarkResult]) -> str:
    """Markdown table, best nDCG@10 first."""
    header = (
        "| Model | Dim | Recall@5 | Recall@10 | nDCG@10 | MRR | Hit@1 | Query p95 (ms) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = sorted(results, key=lambda r: r.scores.ndcg_at_10, reverse=True)
    body = "".join(
        f"| `{r.model}` | {r.dimensions} | {r.scores.recall_at_5:.3f} | "
        f"{r.scores.recall_at_10:.3f} | {r.scores.ndcg_at_10:.3f} | {r.scores.mrr:.3f} | "
        f"{r.scores.hit_at_1:.3f} | {r.query_p95_ms:.1f} |\n"
        for r in rows
    )
    return header + body
