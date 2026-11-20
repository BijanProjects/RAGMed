"""Is one model really better than another, or just lucky on these 1,000 questions?

Recomputes nDCG@10 per question for every model (dense and hybrid), then runs a paired
bootstrap: resample the questions 10,000 times and recompute the gap between two runs. If the
95% interval of the gap includes zero, the difference could be chance. Also breaks scores down
by question type.

Needs the cached document vectors from `python -m ragmed_retrieval benchmark`.
Run from backend/:  python evals/analysis.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ragmed_retrieval.benchmark import CANDIDATE_POOL, document_vectors, top_k
from ragmed_retrieval.chunking import chunk_pairs
from ragmed_retrieval.cli import MODEL_PRESETS, _default_factory
from ragmed_retrieval.corpus import load_medquad
from ragmed_retrieval.evalset import load_eval_set
from ragmed_retrieval.hybrid import BM25Retriever, rrf_fuse
from ragmed_retrieval.metrics import dedupe, ndcg_at_k

MODELS = ["minilm", "qwen3-0.6b", "bge-m3", "medcpt"]
COMPARISONS = [
    ("bge-m3 dense", "qwen3-0.6b dense"),
    ("bge-m3 dense", "minilm dense"),
    ("qwen3-0.6b dense", "minilm dense"),
    ("minilm dense", "medcpt dense"),
    ("bge-m3 hybrid", "bge-m3 dense"),
    ("qwen3-0.6b hybrid", "qwen3-0.6b dense"),
    ("minilm hybrid", "minilm dense"),
    ("medcpt hybrid", "medcpt dense"),
]
RESAMPLES = 10_000
OUT = Path("evals/analysis/significance.json")


def main() -> None:
    pairs, _ = load_medquad(Path("data/raw/medquad.csv"))
    chunks = chunk_pairs(pairs)
    queries = load_eval_set(Path("evals/data/medquad-eval-v1.jsonl"))
    texts = [q.query for q in queries]
    bm25 = BM25Retriever([c.text for c in chunks]).search(texts, CANDIDATE_POOL)

    def per_query_ndcg(ranked: list[list[int]]) -> npt.NDArray[np.float64]:
        return np.array(
            [
                ndcg_at_k(dedupe(chunks[i].parent_id for i in row), q.relevant_ids, 10)
                for row, q in zip(ranked, queries, strict=True)
            ]
        )

    runs = {"bm25 only": per_query_ndcg(bm25)}
    for key in MODELS:
        model_id, prompt = MODEL_PRESETS[key]
        # Best available device: scores do not depend on it, only speed does.
        embedder = _default_factory(model_id, prompt, None)
        docs, _ = document_vectors(embedder, chunks, Path("data/artifacts/embeddings"))
        dense = top_k(embedder.embed_queries(texts), docs, CANDIDATE_POOL).tolist()
        runs[f"{key} dense"] = per_query_ndcg(dense)
        runs[f"{key} hybrid"] = per_query_ndcg(
            [rrf_fuse([d, s]) for d, s in zip(dense, bm25, strict=True)]
        )

    resample = np.random.default_rng(0).integers(0, len(queries), (RESAMPLES, len(queries)))
    gaps = {}
    for a, b in COMPARISONS:
        diff = runs[a] - runs[b]
        low, high = np.percentile(diff[resample].mean(axis=1), [2.5, 97.5])
        gaps[f"{a} vs {b}"] = {
            "gap": round(float(diff.mean()), 4),
            "ci95": [round(float(low), 4), round(float(high), 4)],
            "real": bool(low > 0 or high < 0),
        }

    by_type: dict[str, list[int]] = defaultdict(list)
    for index, query in enumerate(queries):
        by_type[query.qtype].append(index)
    breakdown = {
        qtype: {"n": len(ix)} | {run: round(float(runs[run][ix].mean()), 3) for run in runs}
        for qtype, ix in sorted(by_type.items(), key=lambda item: -len(item[1]))
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "ndcg_at_10": {run: round(float(v.mean()), 4) for run, v in runs.items()},
        "paired_bootstrap": gaps,
        "by_question_type": breakdown,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["paired_bootstrap"], indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
