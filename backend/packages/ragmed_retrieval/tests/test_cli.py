import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from ragmed_retrieval.cli import main
from ragmed_retrieval.embedding import Embedder
from ragmed_retrieval.testing import HashingEmbedder


class FixedReranker:
    def __init__(self, name: str) -> None:
        self.name = name

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        return [0.0] * len(passages)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "medquad.csv"
    rows = [
        ("symptoms", "What are the symptoms of asthma?", "Asthma causes wheezing."),
        ("treatment", "How is diabetes treated?", "Diabetes is treated with insulin."),
        ("causes", "What causes influenza?", "Influenza is caused by a virus."),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qtype", "Question", "Answer"])
        writer.writerows(rows)
    return path


def test_build_eval_writes_jsonl(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "eval.jsonl"

    assert main(["build-eval", "--corpus", str(corpus), "--size", "2", "--out", str(out)]) == 0

    assert len(out.read_text().splitlines()) == 2


def test_benchmark_then_report(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_path, results = tmp_path / "eval.jsonl", tmp_path / "results"
    main(["build-eval", "--corpus", str(corpus), "--size", "3", "--out", str(eval_path)])
    seen: list[tuple[str, str | None, str | None]] = []

    def factory(model_id: str, prompt: str | None, device: str | None) -> Embedder:
        seen.append((model_id, prompt, device))
        return HashingEmbedder()

    code = main(
        [
            "benchmark",
            "--model", "qwen3-0.6b",
            "--corpus", str(corpus),
            "--eval", str(eval_path),
            "--cache", str(tmp_path / "cache"),
            "--out", str(results),
        ],
        factory=factory,
    )  # fmt: skip

    assert code == 0
    assert seen == [("Qwen/Qwen3-Embedding-0.6B", "query", None)]
    saved = json.loads((results / "Qwen__Qwen3-Embedding-0.6B.json").read_text())
    assert saved["scores"]["queries"] == 3

    rerank_code = main(
        [
            "benchmark",
            "--model", "minilm",
            "--corpus", str(corpus),
            "--eval", str(eval_path),
            "--cache", str(tmp_path / "cache"),
            "--out", str(results),
            "--hybrid",
            "--rerank", "bge-reranker-v2-m3",
        ],
        factory=factory,
        reranker_factory=FixedReranker,
    )  # fmt: skip
    assert rerank_code == 0
    saved = json.loads(
        (
            results
            / "sentence-transformers__all-MiniLM-L6-v2--hybrid+rerank--bge-reranker-v2-m3.json"
        ).read_text()
    )
    assert saved["pipeline"] == "hybrid+rerank"

    capsys.readouterr()
    assert main(["report", "--results", str(results)]) == 0
    assert "| `test/hashing` |" in capsys.readouterr().out


def test_report_without_results_fails(tmp_path: Path) -> None:
    assert main(["report", "--results", str(tmp_path)]) == 1


def test_search_prints_the_best_answers(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    used: list[tuple[str, str | None]] = []

    def factory(model_id: str, prompt: str | None, device: str | None) -> Embedder:
        used.append((model_id, device))
        return HashingEmbedder()

    args = ["search", "insulin for diabetes", "--corpus", str(corpus), "--top-k", "2"]
    cache = ["--cache", str(tmp_path / "cache"), "--threads", "1"]

    assert main([*args, *cache], factory=factory) == 0
    first = capsys.readouterr().out
    assert main([*args, *cache], factory=factory) == 0
    second = capsys.readouterr().out

    assert used[0] == ("BAAI/bge-m3", "cpu")
    assert "embedded 3 passages" in first
    assert "embedded" not in second
    lines = [line for line in second.splitlines() if line.strip().startswith("1.")]
    assert "How is diabetes treated?" in lines[0]
    assert "  3." not in second
