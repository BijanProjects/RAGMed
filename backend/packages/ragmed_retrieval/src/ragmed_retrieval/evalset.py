"""Build and persist the retrieval evaluation set.

Each query is a MedQuAD question; its relevant documents are every answer given for that
question. Only answers are indexed, so a query never matches its own text verbatim.
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ragmed_retrieval.corpus import QAPair, make_id


@dataclass(frozen=True, slots=True)
class EvalQuery:
    id: str
    query: str
    qtype: str
    relevant_ids: tuple[str, ...]


def build_eval_set(pairs: Iterable[QAPair], size: int, seed: int) -> list[EvalQuery]:
    """Sample `size` distinct questions deterministically for a given seed."""
    by_question: dict[str, list[QAPair]] = defaultdict(list)
    for pair in pairs:
        by_question[pair.question].append(pair)

    questions = sorted(by_question)
    if not 0 < size <= len(questions):
        raise ValueError(f"size must be between 1 and {len(questions)}")

    sample = random.Random(seed).sample(questions, size)  # noqa: S311 - reproducible sampling, not security
    return [
        EvalQuery(
            id=make_id(question, "eval"),
            query=question,
            qtype=by_question[question][0].qtype,
            relevant_ids=tuple(sorted(p.id for p in by_question[question])),
        )
        for question in sample
    ]


def save_eval_set(queries: Iterable[EvalQuery], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")


def load_eval_set(path: Path) -> list[EvalQuery]:
    with path.open(encoding="utf-8") as handle:
        return [
            EvalQuery(
                id=row["id"],
                query=row["query"],
                qtype=row["qtype"],
                relevant_ids=tuple(row["relevant_ids"]),
            )
            for row in map(json.loads, handle)
        ]
