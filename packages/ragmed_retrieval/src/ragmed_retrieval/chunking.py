"""Sentence-aware chunking of answers into retrievable passages.

Word counts stand in for tokens (about 0.75 words per token for English medical text),
which keeps chunking independent of whichever embedding model is chosen.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ragmed_retrieval.corpus import QAPair

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

DEFAULT_MAX_WORDS = 384
DEFAULT_OVERLAP_WORDS = 48


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    parent_id: str
    index: int
    qtype: str
    question: str
    text: str


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s]


def _word_count(sentence: str) -> int:
    return len(sentence.split())


def _hard_split(sentence: str, max_words: int, overlap_words: int) -> list[str]:
    """Split one over-long sentence into overlapping word windows."""
    words = sentence.split()
    step = max_words - overlap_words
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words) - overlap_words, step)]


def _overlap_tail(sentences: list[str], overlap_words: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        total += _word_count(sentence)
        if total > overlap_words:
            break
        tail.insert(0, sentence)
    return tail


def chunk_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words")

    pieces: list[str] = []
    for sentence in split_sentences(text):
        if _word_count(sentence) > max_words:
            pieces.extend(_hard_split(sentence, max_words, overlap_words))
        else:
            pieces.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for piece in pieces:
        words = _word_count(piece)
        if current and current_words + words > max_words:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap_words)
            current_words = sum(_word_count(s) for s in current)
        current.append(piece)
        current_words += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pairs(
    pairs: Iterable[QAPair],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk each answer. Every chunk keeps its parent id for citations."""
    return [
        Chunk(f"{pair.id}:{index}", pair.id, index, pair.qtype, pair.question, text)
        for pair in pairs
        for index, text in enumerate(chunk_text(pair.answer, max_words, overlap_words))
    ]
