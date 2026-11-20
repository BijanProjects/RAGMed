"""Command-line entry points: `python -m ragmed_retrieval <command>`."""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from ragmed_retrieval.benchmark import (
    BenchmarkResult,
    Pipeline,
    document_vectors,
    result_filename,
    results_table,
    run_benchmark,
    top_k,
)
from ragmed_retrieval.chunking import chunk_pairs
from ragmed_retrieval.corpus import load_medquad
from ragmed_retrieval.embedding import (
    MEDCPT,
    Embedder,
    SentenceTransformerEmbedder,
    medcpt_embedder,
)
from ragmed_retrieval.evalset import build_eval_set, load_eval_set, save_eval_set
from ragmed_retrieval.hybrid import RERANKER_PRESETS, BM25Retriever, Reranker
from ragmed_retrieval.metrics import RetrievalScores, dedupe

# Short name -> (Hugging Face id, query prompt name)
MODEL_PRESETS: dict[str, tuple[str, str | None]] = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", None),
    "bge-m3": ("BAAI/bge-m3", None),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "query"),
    "qwen3-4b": ("Qwen/Qwen3-Embedding-4B", "query"),
    "embeddinggemma": ("google/embeddinggemma-300m", "query"),
    "medcpt": (MEDCPT, None),
}

# Chosen by the benchmark: tied with Qwen3-Embedding-0.6B on accuracy, faster on CPU.
# See ../README.md#embedding-benchmark.
DEFAULT_MODEL = "bge-m3"

EmbedderFactory = Callable[[str, str | None, str | None], Embedder]
RerankerFactory = Callable[[str], Reranker]


def _default_factory(  # pragma: no cover
    model_id: str, query_prompt: str | None, device: str | None
) -> Embedder:
    if model_id == MEDCPT:
        return medcpt_embedder(device)
    return SentenceTransformerEmbedder(model_id, query_prompt_name=query_prompt, device=device)


def _limit_threads(threads: int | None) -> None:  # pragma: no cover - needs torch
    """Cap PyTorch's CPU threads, e.g. to 4 to match the 4-core Oracle server."""
    if threads is None:
        return
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # no PyTorch installed, so nothing to limit
        return
    torch.set_num_threads(threads)


def _resolve(model: str, query_prompt: str | None = None) -> tuple[str, str | None]:
    return MODEL_PRESETS.get(model, (model, query_prompt))


def _default_reranker(name: str) -> Reranker:  # pragma: no cover
    return RERANKER_PRESETS[name]()


def _build_eval(args: argparse.Namespace) -> int:
    pairs, report = load_medquad(args.corpus)
    queries = build_eval_set(pairs, size=args.size, seed=args.seed)
    save_eval_set(queries, args.out)
    print(f"{len(queries)} queries from {report.kept} answers -> {args.out}")
    return 0


def _benchmark(
    args: argparse.Namespace, factory: EmbedderFactory, reranker_factory: RerankerFactory
) -> int:
    model_id, query_prompt = _resolve(args.model, args.query_prompt)
    _limit_threads(args.threads)
    pairs, _ = load_medquad(args.corpus)
    chunks = chunk_pairs(pairs)
    queries = load_eval_set(args.eval)

    pipeline = Pipeline(
        sparse=BM25Retriever([c.text for c in chunks]) if args.hybrid else None,
        reranker=reranker_factory(args.rerank) if args.rerank else None,
    )
    embedder = factory(model_id, query_prompt, args.device)

    result = run_benchmark(embedder, chunks, queries, pipeline=pipeline, cache_dir=args.cache)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / result_filename(result, model_id)
    path.write_text(result.to_json() + "\n", encoding="utf-8")
    print(f"{model_id} [{result.pipeline}]: nDCG@10={result.scores.ndcg_at_10:.3f} -> {path}")
    return 0


def _search(args: argparse.Namespace, factory: EmbedderFactory) -> int:
    model_id, query_prompt = _resolve(args.model)
    _limit_threads(args.threads)
    pairs, _ = load_medquad(args.corpus)
    chunks = chunk_pairs(pairs)
    embedder = factory(model_id, query_prompt, args.device)
    doc_vectors, seconds = document_vectors(embedder, chunks, args.cache)
    if seconds:
        print(f"embedded {len(chunks)} passages in {seconds:.0f} s (saved for next time)")
    embedder.embed_queries(["warm-up"])  # the first call pays one-off setup; keep it out of timings

    for question in args.questions:
        start = time.perf_counter()
        query_vector = embedder.embed_queries([question])
        ranked = top_k(query_vector, doc_vectors, args.top_k * 4)[0].tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000
        best: dict[str, tuple[float, int]] = {}
        for index in ranked:
            parent = chunks[index].parent_id
            if parent not in best:
                best[parent] = (float(query_vector[0] @ doc_vectors[index]), index)
        print(f"\nQ: {question}\n   {model_id} on {args.device or 'auto'}: {elapsed_ms:.0f} ms")
        for rank, parent in enumerate(dedupe(chunks[i].parent_id for i in ranked)[: args.top_k], 1):
            score, index = best[parent]
            chunk = chunks[index]
            snippet = " ".join(chunk.text.split())[: args.width]
            print(f"{rank:>3}. [{score:.3f}] {chunk.qtype} | {chunk.question}\n      {snippet}...")
    return 0


def _load_result(path: Path) -> BenchmarkResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["scores"] = RetrievalScores(**raw["scores"])
    return BenchmarkResult(**raw)


def _report(args: argparse.Namespace) -> int:
    results = [_load_result(p) for p in sorted(args.results.glob("*.json"))]
    if not results:
        print(f"no results in {args.results}")
        return 1
    print(results_table(results), end="")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragmed_retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-eval", help="sample the retrieval evaluation set")
    build.add_argument("--corpus", type=Path, default=Path("data/raw/medquad.csv"))
    build.add_argument("--size", type=int, default=1000)
    build.add_argument("--seed", type=int, default=2026)
    build.add_argument("--out", type=Path, required=True)

    bench = sub.add_parser("benchmark", help="score one embedding model")
    bench.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"preset {sorted(MODEL_PRESETS)} or HF id"
    )
    bench.add_argument("--query-prompt", default=None, help="prompt name for non-preset models")
    bench.add_argument("--corpus", type=Path, default=Path("data/raw/medquad.csv"))
    bench.add_argument("--eval", type=Path, required=True)
    bench.add_argument("--cache", type=Path, default=Path("data/artifacts/embeddings"))
    bench.add_argument("--out", type=Path, default=Path("evals/results"))
    bench.add_argument("--hybrid", action="store_true", help="fuse dense with BM25 (RRF)")
    bench.add_argument("--rerank", choices=sorted(RERANKER_PRESETS), help="rerank the top 40")
    bench.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    bench.add_argument("--threads", type=int, default=None, help="CPU threads (Oracle server: 4)")

    search = sub.add_parser("search", help="ask questions against the knowledge base")
    search.add_argument("questions", nargs="+", help="one or more questions, each in quotes")
    search.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_PRESETS))
    search.add_argument("--device", default="cpu", help="cpu (as on the server), mps or cuda")
    search.add_argument("--threads", type=int, default=4, help="CPU threads (Oracle server: 4)")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--width", type=int, default=160, help="characters of each passage shown")
    search.add_argument("--corpus", type=Path, default=Path("data/raw/medquad.csv"))
    search.add_argument("--cache", type=Path, default=Path("data/artifacts/embeddings"))

    report = sub.add_parser("report", help="print a markdown table of saved results")
    report.add_argument("--results", type=Path, default=Path("evals/results"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    factory: EmbedderFactory = _default_factory,
    reranker_factory: RerankerFactory = _default_reranker,
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-eval":
        return _build_eval(args)
    if args.command == "benchmark":
        return _benchmark(args, factory, reranker_factory)
    if args.command == "search":
        return _search(args, factory)
    return _report(args)
