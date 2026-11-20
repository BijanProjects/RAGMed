"""Is one setup really better than another, or just lucky on these 1,000 questions?

Reads the per-question nDCG@10 scores that `python -m ragmed_retrieval benchmark` saves in
evals/results/per-query/, then runs a paired bootstrap: resample the questions 10,000 times
and recompute the gap between two runs. If the 95% interval of the gap includes zero, the
difference could be chance. Also breaks scores down by question type.

Run from backend/:  python evals/analysis.py   (or: make significance)
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ragmed_retrieval.benchmark import CANDIDATE_POOL
from ragmed_retrieval.chunking import chunk_pairs
from ragmed_retrieval.corpus import load_medquad
from ragmed_retrieval.evalset import load_eval_set
from ragmed_retrieval.hybrid import BM25Retriever
from ragmed_retrieval.metrics import dedupe, ndcg_at_k

PER_QUERY = Path("evals/results/per-query")
OUT = Path("evals/analysis/significance.json")
RESAMPLES = 10_000

# Short name -> result file (as written by the benchmark command)
RUNS = {
    "minilm dense": "sentence-transformers__all-MiniLM-L6-v2.json",
    "minilm hybrid": "sentence-transformers__all-MiniLM-L6-v2--hybrid.json",
    "qwen3-0.6b dense": "Qwen__Qwen3-Embedding-0.6B.json",
    "qwen3-0.6b hybrid": "Qwen__Qwen3-Embedding-0.6B--hybrid.json",
    "bge-m3 dense": "BAAI__bge-m3.json",
    "bge-m3 hybrid": "BAAI__bge-m3--hybrid.json",
    "medcpt dense": "ncbi__MedCPT.json",
    "medcpt hybrid": "ncbi__MedCPT--hybrid.json",
    "bge-m3 onnx-int8": "BAAI__bge-m3--onnx-int8.json",
    "bge-m3 + minilm-ce top40": "BAAI__bge-m3--dense+rerank--ms-marco-MiniLM-L6-v2.json",
    "bge-m3 + minilm-ce top20": "BAAI__bge-m3--dense+rerank--ms-marco-MiniLM-L6-v2--top20.json",
    "bge-m3 + bge-reranker-base": "BAAI__bge-m3--dense+rerank--bge-reranker-base.json",
    "bge-m3 + bge-reranker-v2-m3": "BAAI__bge-m3--dense+rerank--bge-reranker-v2-m3.json",
}
COMPARISONS = [
    ("bge-m3 dense", "qwen3-0.6b dense"),
    ("bge-m3 dense", "minilm dense"),
    ("qwen3-0.6b dense", "minilm dense"),
    ("minilm dense", "medcpt dense"),
    ("bge-m3 hybrid", "bge-m3 dense"),
    ("qwen3-0.6b hybrid", "qwen3-0.6b dense"),
    ("minilm hybrid", "minilm dense"),
    ("medcpt hybrid", "medcpt dense"),
    ("bge-m3 onnx-int8", "bge-m3 dense"),
    ("bge-m3 + minilm-ce top40", "bge-m3 dense"),
    ("bge-m3 + minilm-ce top20", "bge-m3 dense"),
    ("bge-m3 + bge-reranker-base", "bge-m3 dense"),
    ("bge-m3 + bge-reranker-v2-m3", "bge-m3 dense"),
]


def main() -> None:
    queries = load_eval_set(Path("evals/data/medquad-eval-v1.jsonl"))
    ids = [q.id for q in queries]

    runs: dict[str, npt.NDArray[np.float64]] = {}
    for name, filename in RUNS.items():
        path = PER_QUERY / filename
        if path.exists():
            scores = json.loads(path.read_text(encoding="utf-8"))
            runs[name] = np.array([scores[i] for i in ids])
        else:
            print(f"skipping {name}: no {path}")

    # Keyword search alone needs no model, so it is computed here.
    chunks = chunk_pairs(load_medquad(Path("data/raw/medquad.csv"))[0])
    bm25 = BM25Retriever([c.text for c in chunks]).search([q.query for q in queries], 100)
    runs["bm25 only"] = np.array(
        [
            ndcg_at_k(dedupe(chunks[i].parent_id for i in row[:CANDIDATE_POOL]), q.relevant_ids, 10)
            for row, q in zip(bm25, queries, strict=True)
        ]
    )

    resample = np.random.default_rng(0).integers(0, len(queries), (RESAMPLES, len(queries)))
    gaps = {}
    for a, b in COMPARISONS:
        if a not in runs or b not in runs:
            continue
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
    for name, gap in gaps.items():
        low, high = gap["ci95"]
        verdict = "real" if gap["real"] else "could be chance"
        print(f"{name:55s} {gap['gap']:+.4f}  [{low:+.4f}, {high:+.4f}]  {verdict}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
