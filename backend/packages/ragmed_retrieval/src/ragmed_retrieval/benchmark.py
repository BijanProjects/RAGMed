"""Retrieval benchmark: embed the corpus once, then score the evaluation set.

Three pipelines are compared: dense only, hybrid (dense + BM25 fused with RRF), and either
of those followed by a reranker over the top `rerank_depth` candidates.
"""

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
from ragmed_retrieval.hybrid import Reranker, SparseRetriever, rerank_head, rrf_fuse
from ragmed_retrieval.metrics import RetrievalScores, dedupe, score_run

CANDIDATE_POOL = 100
RERANK_DEPTH = 40
LATENCY_SAMPLE = 100


@dataclass(frozen=True, slots=True)
class Pipeline:
    sparse: SparseRetriever | None = None
    reranker: Reranker | None = None
    rerank_depth: int = RERANK_DEPTH

    @property
    def label(self) -> str:
        base = "hybrid" if self.sparse is not None else "dense"
        return f"{base}+rerank" if self.reranker is not None else base


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model: str
    chunks: int
    dimensions: int
    index_seconds: float
    query_p50_ms: float
    query_p95_ms: float
    scores: RetrievalScores
    pipeline: str = "dense"
    reranker: str | None = None

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


def document_vectors(
    embedder: Embedder, chunks: Sequence[Chunk], cache_dir: Path | None
) -> tuple[Vectors, float]:
    """Embed every chunk, or load the vectors saved by an earlier run; also returns seconds."""
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


def _retrieve(
    embedder: Embedder,
    texts: Sequence[str],
    doc_vectors: Vectors,
    chunks: Sequence[Chunk],
    pipeline: Pipeline,
) -> list[list[int]]:
    """Ranked chunk indices per query text."""
    dense = top_k(embedder.embed_queries(texts), doc_vectors, CANDIDATE_POOL).tolist()
    if pipeline.sparse is None:
        ranked: list[list[int]] = dense
    else:
        sparse = pipeline.sparse.search(texts, CANDIDATE_POOL)
        ranked = [rrf_fuse([d, s]) for d, s in zip(dense, sparse, strict=True)]
    if pipeline.reranker is not None:
        passages = [c.text for c in chunks]
        ranked = [
            rerank_head(text, row, passages, pipeline.reranker, pipeline.rerank_depth)
            for text, row in zip(texts, ranked, strict=True)
        ]
    return ranked


def _query_latency_ms(
    retrieve_one: Callable[[str], object],
    queries: Sequence[EvalQuery],
    clock: Callable[[], float],
) -> tuple[float, float]:
    """Wall time of the whole pipeline for one query at a time, as a live request sees it."""
    timings = []
    for query in queries[:LATENCY_SAMPLE]:
        start = clock()
        retrieve_one(query.query)
        timings.append((clock() - start) * 1000)
    p50, p95 = np.percentile(timings, [50, 95])
    return float(p50), float(p95)


def run_benchmark(
    embedder: Embedder,
    chunks: Sequence[Chunk],
    queries: Sequence[EvalQuery],
    *,
    pipeline: Pipeline | None = None,
    cache_dir: Path | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    pipeline = pipeline or Pipeline()
    doc_vectors, index_seconds = document_vectors(embedder, chunks, cache_dir)

    ranked = _retrieve(embedder, [q.query for q in queries], doc_vectors, chunks, pipeline)
    results = [
        (dedupe(chunks[i].parent_id for i in row), query.relevant_ids)
        for row, query in zip(ranked, queries, strict=True)
    ]
    p50, p95 = _query_latency_ms(
        lambda text: _retrieve(embedder, [text], doc_vectors, chunks, pipeline), queries, clock
    )

    return BenchmarkResult(
        model=embedder.name,
        chunks=len(chunks),
        dimensions=int(doc_vectors.shape[1]),
        index_seconds=round(index_seconds, 1),
        query_p50_ms=round(p50, 1),
        query_p95_ms=round(p95, 1),
        scores=score_run(results),
        pipeline=pipeline.label,
        reranker=pipeline.reranker.name if pipeline.reranker is not None else None,
    )


def results_table(results: Sequence[BenchmarkResult]) -> str:
    """Markdown table, best nDCG@10 first."""
    header = (
        "| Model | Pipeline | Dim | Recall@5 | Recall@10 | nDCG@10 | MRR | Hit@1 "
        "| Query p95 (ms) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = sorted(results, key=lambda r: r.scores.ndcg_at_10, reverse=True)
    body = "".join(
        f"| `{r.model}` | {_pipeline_cell(r)} | {r.dimensions} | {r.scores.recall_at_5:.3f} | "
        f"{r.scores.recall_at_10:.3f} | {r.scores.ndcg_at_10:.3f} | {r.scores.mrr:.3f} | "
        f"{r.scores.hit_at_1:.3f} | {r.query_p95_ms:.1f} |\n"
        for r in rows
    )
    return header + body


def _pipeline_cell(result: BenchmarkResult) -> str:
    if result.reranker is None:
        return result.pipeline
    return f"{result.pipeline} (`{result.reranker.rsplit('/', 1)[-1]}`)"


def result_filename(result: BenchmarkResult, model_id: str | None = None) -> str:
    """File name for a saved result; `model_id` overrides the embedder's own name."""
    slug = (model_id or result.model).replace("/", "__")
    if result.pipeline != "dense":
        slug += f"--{result.pipeline}"
    if result.reranker is not None:
        slug += f"--{result.reranker.rsplit('/', 1)[-1]}"
    return f"{slug}.json"
