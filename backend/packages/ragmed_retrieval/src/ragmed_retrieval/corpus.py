"""Load and clean the MedQuAD question-answer corpus."""

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")
# MedQuAD questions often end in a doubled mark, e.g. "Who is at risk for LCM? ?"
_REPEATED_QMARK = re.compile(r"\s*\?(?:\s*\?)*")

REQUIRED_COLUMNS = frozenset({"qtype", "Question", "Answer"})


@dataclass(frozen=True, slots=True)
class QAPair:
    id: str
    qtype: str
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class LoadReport:
    rows_read: int
    empty_dropped: int
    duplicates_dropped: int

    @property
    def kept(self) -> int:
        return self.rows_read - self.empty_dropped - self.duplicates_dropped


def normalize_text(text: str) -> str:
    text = _WHITESPACE.sub(" ", text).strip()
    return _REPEATED_QMARK.sub("?", text)


def make_id(question: str, answer: str) -> str:
    """Stable id from content, so re-ingesting the same corpus yields the same ids."""
    digest = hashlib.sha256(f"{question}\n{answer}".encode()).hexdigest()
    return digest[:16]


def load_medquad(path: Path) -> tuple[list[QAPair], LoadReport]:
    """Read the corpus, normalise whitespace, drop empty rows and exact duplicates."""
    pairs: list[QAPair] = []
    seen: set[str] = set()
    rows_read = empty = duplicates = 0

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            rows_read += 1
            question = normalize_text(row["Question"] or "")
            answer = normalize_text(row["Answer"] or "")
            if not question or not answer:
                empty += 1
                continue
            pair_id = make_id(question, answer)
            if pair_id in seen:
                duplicates += 1
                continue
            seen.add(pair_id)
            qtype = normalize_text(row["qtype"] or "").lower() or "unknown"
            pairs.append(QAPair(pair_id, qtype, question, answer))

    return pairs, LoadReport(rows_read, empty, duplicates)
