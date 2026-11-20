import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from ragmed_retrieval.cli import main
from ragmed_retrieval.embedding import Embedder
from ragmed_retrieval.testing import HashingEmbedder, write_tiny_onnx_model


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
    assert "per_query" not in saved
    per_query = json.loads((results / "per-query" / "Qwen__Qwen3-Embedding-0.6B.json").read_text())
    assert len(per_query) == 3
    assert all(0.0 <= v <= 1.0 for v in per_query.values())

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
            "--rerank-depth", "2",
        ],
        factory=factory,
        reranker_factory=FixedReranker,
    )  # fmt: skip
    assert rerank_code == 0
    name = "sentence-transformers__all-MiniLM-L6-v2--hybrid+rerank--bge-reranker-v2-m3--top2.json"
    saved = json.loads((results / name).read_text())
    assert saved["pipeline"] == "hybrid+rerank"
    assert saved["rerank_depth"] == 2

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
    cache = ["--cache", str(tmp_path / "cache"), "--threads", "1", "--onnx-dir", str(tmp_path)]

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


def test_search_uses_the_onnx_export_when_present(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    onnx_root = tmp_path / "onnx"
    write_tiny_onnx_model(onnx_root / "BAAI__bge-m3", "BAAI/bge-m3")

    def no_torch(model_id: str, prompt: str | None, device: str | None) -> Embedder:
        raise AssertionError("the ONNX export should be used")

    # The tiny model only looks at the first word, so the question starts with the disease.
    args = ["search", "diabetes and insulin", "--corpus", str(corpus), "--top-k", "1"]
    paths = ["--cache", str(tmp_path / "cache"), "--onnx-dir", str(onnx_root)]

    assert main([*args, *paths], factory=no_torch) == 0
    out = capsys.readouterr().out
    assert "BAAI/bge-m3 (onnx-int8, cpu)" in out
    assert "How is diabetes treated?" in out


def test_onnx_runtime_without_an_export_explains_what_to_do(corpus: Path, tmp_path: Path) -> None:
    args = ["search", "q", "--corpus", str(corpus), "--runtime", "onnx"]

    with pytest.raises(SystemExit, match="make export-onnx"):
        main([*args, "--onnx-dir", str(tmp_path)], factory=lambda *_: HashingEmbedder())


def test_benchmark_records_the_onnx_runtime(corpus: Path, tmp_path: Path) -> None:
    eval_path, results, onnx_root = tmp_path / "eval.jsonl", tmp_path / "res", tmp_path / "onnx"
    main(["build-eval", "--corpus", str(corpus), "--size", "3", "--out", str(eval_path)])
    write_tiny_onnx_model(onnx_root / "BAAI__bge-m3", "BAAI/bge-m3")

    code = main(
        [
            "benchmark", "--corpus", str(corpus), "--eval", str(eval_path),
            "--cache", str(tmp_path / "cache"), "--out", str(results),
            "--runtime", "onnx", "--onnx-dir", str(onnx_root),
        ]
    )  # fmt: skip

    assert code == 0
    saved = json.loads((results / "BAAI__bge-m3--onnx-int8.json").read_text())
    assert saved["runtime"] == "onnx-int8"
