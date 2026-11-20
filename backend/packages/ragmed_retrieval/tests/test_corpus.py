import csv
from pathlib import Path

import pytest

from ragmed_retrieval.corpus import QAPair, load_medquad, make_id, normalize_text


def _write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


FIELDS = ("qtype", "Question", "Answer")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Who is at risk for LCM? ?", "Who is at risk for LCM?"),
        ("What causes\n\n  asthma ?? ?", "What causes asthma?"),
        ("tabs\tand   spaces", "tabs and spaces"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_ids_are_stable_and_content_based() -> None:
    assert make_id("q", "a") == make_id("q", "a")
    assert make_id("q", "a") != make_id("q", "b")
    assert len(make_id("q", "a")) == 16


def test_load_cleans_dedupes_and_reports(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "medquad.csv",
        [
            {"qtype": "Symptoms", "Question": "Symptoms of flu? ?", "Answer": "Fever."},
            {"qtype": "symptoms", "Question": "Symptoms of flu?", "Answer": " Fever. "},
            {"qtype": "treatment", "Question": "How is flu treated?", "Answer": ""},
            {"qtype": "", "Question": "What is flu?", "Answer": "A viral."},
        ],
        FIELDS,
    )

    pairs, report = load_medquad(path)

    assert report.rows_read == 4
    assert report.empty_dropped == 1
    assert report.duplicates_dropped == 1
    assert report.kept == 2
    assert pairs == [
        QAPair(make_id("Symptoms of flu?", "Fever."), "symptoms", "Symptoms of flu?", "Fever."),
        QAPair(make_id("What is flu?", "A viral."), "unknown", "What is flu?", "A viral."),
    ]


def test_load_rejects_missing_columns(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "bad.csv", [{"Question": "q", "Answer": "a"}], FIELDS[1:])

    with pytest.raises(ValueError, match="qtype"):
        load_medquad(path)
