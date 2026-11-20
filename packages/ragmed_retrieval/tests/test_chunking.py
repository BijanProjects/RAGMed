from itertools import pairwise

import pytest

from ragmed_retrieval.chunking import chunk_pairs, chunk_text, split_sentences
from ragmed_retrieval.corpus import QAPair


def _sentence(n: int, words: int = 10) -> str:
    return " ".join([f"S{n}"] + ["word"] * (words - 2) + ["end."])


def test_split_sentences_keeps_abbreviation_like_text_together() -> None:
    text = "Fever is common. Take 500 mg (e.g. paracetamol). 3 doses a day! Done?"

    assert split_sentences(text) == [
        "Fever is common.",
        "Take 500 mg (e.g. paracetamol).",
        "3 doses a day!",
        "Done?",
    ]


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("One sentence. Two sentences.") == ["One sentence. Two sentences."]


def test_long_text_is_split_within_budget_with_overlap() -> None:
    text = " ".join(_sentence(i) for i in range(10))  # 100 words

    chunks = chunk_text(text, max_words=30, overlap_words=10)

    assert len(chunks) > 1
    assert all(len(c.split()) <= 30 for c in chunks)
    for previous, nxt in pairwise(chunks):
        assert previous.split(". ")[-1] in nxt  # last sentence carried over
    assert chunks[0].startswith("S0 ")
    assert chunks[-1].endswith(_sentence(9))


def test_every_sentence_is_covered_in_order() -> None:
    sentences = [_sentence(i, words=7) for i in range(25)]

    chunks = chunk_text(" ".join(sentences), max_words=40, overlap_words=0)

    assert " ".join(chunks) == " ".join(sentences)


def test_over_long_sentence_is_hard_split() -> None:
    sentence = " ".join(f"w{i}" for i in range(100)) + "."

    chunks = chunk_text(sentence, max_words=40, overlap_words=10)

    assert all(len(c.split()) <= 40 for c in chunks)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w99."


@pytest.mark.parametrize(("max_words", "overlap"), [(10, 10), (10, -1)])
def test_invalid_overlap_is_rejected(max_words: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_text("text.", max_words=max_words, overlap_words=overlap)


def test_chunk_pairs_keeps_parent_metadata() -> None:
    long_answer = " ".join(_sentence(i) for i in range(10))
    pairs = [
        QAPair("aaa", "symptoms", "What are the symptoms?", "Short answer."),
        QAPair("bbb", "treatment", "How is it treated?", long_answer),
    ]

    chunks = chunk_pairs(pairs, max_words=30, overlap_words=10)

    assert chunks[0].id == "aaa:0"
    assert chunks[0].text == "Short answer."
    treatment = [c for c in chunks if c.parent_id == "bbb"]
    assert [c.index for c in treatment] == list(range(len(treatment)))
    assert all(c.question == "How is it treated?" and c.qtype == "treatment" for c in treatment)


def test_overlap_never_pushes_a_chunk_over_budget() -> None:
    text = " ".join([_sentence(0, words=8), _sentence(1, words=8), _sentence(2, words=28)])

    chunks = chunk_text(text, max_words=30, overlap_words=10)

    assert all(len(c.split()) <= 30 for c in chunks)
