"""Command-line entry points: `python -m ragmed_retrieval <command>`."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from ragmed_retrieval.benchmark import BenchmarkResult, results_table, run_benchmark
from ragmed_retrieval.chunking import chunk_pairs
from ragmed_retrieval.corpus import load_medquad
from ragmed_retrieval.embedding import Embedder, SentenceTransformerEmbedder
from ragmed_retrieval.evalset import build_eval_set, load_eval_set, save_eval_set
from ragmed_retrieval.metrics import RetrievalScores

# Short name -> (Hugging Face id, query prompt name)
MODEL_PRESETS: dict[str, tuple[str, str | None]] = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", None),
    "bge-m3": ("BAAI/bge-m3", None),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "query"),
    "qwen3-4b": ("Qwen/Qwen3-Embedding-4B", "query"),
    "embeddinggemma": ("google/embeddinggemma-300m", "query"),
}

EmbedderFactory = Callable[[str, str | None], Embedder]


def _default_factory(model_id: str, query_prompt: str | None) -> Embedder:
    return SentenceTransformerEmbedder(model_id, query_prompt_name=query_prompt)


def _build_eval(args: argparse.Namespace) -> int:
    pairs, report = load_medquad(args.corpus)
    queries = build_eval_set(pairs, size=args.size, seed=args.seed)
    save_eval_set(queries, args.out)
    print(f"{len(queries)} queries from {report.kept} answers -> {args.out}")
    return 0


def _benchmark(args: argparse.Namespace, factory: EmbedderFactory) -> int:
    model_id, query_prompt = MODEL_PRESETS.get(args.model, (args.model, args.query_prompt))
    pairs, _ = load_medquad(args.corpus)
    chunks = chunk_pairs(pairs)
    queries = load_eval_set(args.eval)

    result = run_benchmark(factory(model_id, query_prompt), chunks, queries, cache_dir=args.cache)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{model_id.replace('/', '__')}.json"
    path.write_text(result.to_json() + "\n", encoding="utf-8")
    print(f"{model_id}: nDCG@10={result.scores.ndcg_at_10:.3f} -> {path}")
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
    bench.add_argument("--model", required=True, help=f"preset {sorted(MODEL_PRESETS)} or HF id")
    bench.add_argument("--query-prompt", default=None, help="prompt name for non-preset models")
    bench.add_argument("--corpus", type=Path, default=Path("data/raw/medquad.csv"))
    bench.add_argument("--eval", type=Path, required=True)
    bench.add_argument("--cache", type=Path, default=Path("data/artifacts/embeddings"))
    bench.add_argument("--out", type=Path, default=Path("evals/results"))

    report = sub.add_parser("report", help="print a markdown table of saved results")
    report.add_argument("--results", type=Path, default=Path("evals/results"))
    return parser


def main(argv: Sequence[str] | None = None, factory: EmbedderFactory = _default_factory) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-eval":
        return _build_eval(args)
    if args.command == "benchmark":
        return _benchmark(args, factory)
    return _report(args)
