from pathlib import Path

import pytest

from ragmed_retrieval.corpus import QAPair
from ragmed_retrieval.evalset import build_eval_set, load_eval_set, save_eval_set

PAIRS = [
    QAPair("a1", "symptoms", "What are the symptoms of A?", "Fever."),
    QAPair("a2", "symptoms", "What are the symptoms of A?", "Cough."),
    QAPair("b1", "treatment", "How is B treated?", "Rest."),
    QAPair("c1", "causes", "What causes C?", "A virus."),
]


def test_all_answers_to_a_question_are_relevant() -> None:
    queries = build_eval_set(PAIRS, size=3, seed=0)

    by_query = {q.query: q for q in queries}
    assert by_query["What are the symptoms of A?"].relevant_ids == ("a1", "a2")
    assert by_query["How is B treated?"].relevant_ids == ("b1",)
    assert by_query["How is B treated?"].qtype == "treatment"


def test_sampling_is_deterministic_and_distinct() -> None:
    first = build_eval_set(PAIRS, size=2, seed=7)
    again = build_eval_set(list(reversed(PAIRS)), size=2, seed=7)

    assert first == again
    assert len({q.query for q in first}) == 2


@pytest.mark.parametrize("size", [0, 4])
def test_size_must_fit_the_corpus(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        build_eval_set(PAIRS, size=size, seed=0)


def test_round_trip(tmp_path: Path) -> None:
    queries = build_eval_set(PAIRS, size=3, seed=1)
    path = tmp_path / "evals" / "set.jsonl"

    save_eval_set(queries, path)

    assert load_eval_set(path) == queries
