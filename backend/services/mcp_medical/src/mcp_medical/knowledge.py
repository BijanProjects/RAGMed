"""Search the knowledge base: embed the question, query the vector store, one hit per answer."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import anyio.to_thread
import numpy as np
import numpy.typing as npt

from ragmed_retrieval.chunking import Chunk
from ragmed_retrieval.corpus import QAPair
from ragmed_retrieval.embedding import Embedder
from ragmed_vectorstores import Document, Query, VectorStore

# Several passages can come from one long answer; fetch extra so `top_k` distinct answers remain.
OVERFETCH = 3
SOURCE = "MedQuAD"


@dataclass(frozen=True, slots=True)
class Passage:
    id: str
    answer_id: str
    question: str
    qtype: str
    text: str
    score: float
    source: str = SOURCE


class KnowledgeBase:
    def __init__(
        self, embedder: Embedder, store: VectorStore, answers: Mapping[str, QAPair]
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._answers = answers
        self._topics = dict(Counter(a.qtype for a in answers.values()).most_common())

    @property
    def topics(self) -> dict[str, int]:
        """Question types and how many answers each has."""
        return dict(self._topics)

    @property
    def size(self) -> int:
        return len(self._answers)

    def answer(self, answer_id: str) -> QAPair | None:
        return self._answers.get(answer_id)

    async def search(self, query: str, top_k: int = 5, qtype: str | None = None) -> list[Passage]:
        # Embedding is CPU work (about 10 ms); keep it off the event loop.
        vector = await anyio.to_thread.run_sync(self._embedder.embed_queries, [query])
        hits = await self._store.query(
            Query(
                vector[0].tolist(),
                top_k=top_k * OVERFETCH,
                filter={"qtype": qtype} if qtype else None,
            )
        )
        passages: dict[str, Passage] = {}
        for hit in hits:
            answer_id = str(hit.metadata["answer_id"])
            if answer_id not in passages:
                passages[answer_id] = Passage(
                    id=hit.id,
                    answer_id=answer_id,
                    question=str(hit.metadata["question"]),
                    qtype=str(hit.metadata["qtype"]),
                    text=hit.text,
                    score=round(hit.score, 4),
                )
        return list(passages.values())[:top_k]


async def index_chunks(
    store: VectorStore,
    chunks: Sequence[Chunk],
    vectors: npt.NDArray[np.float32],
    *,
    batch_size: int = 256,
) -> int:
    """Load passages and their vectors into the store; safe to run again (upserts by id)."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    for start in range(0, len(chunks), batch_size):
        await store.upsert(
            [
                Document(
                    chunk.id,
                    chunk.text,
                    vector.tolist(),
                    {
                        "answer_id": chunk.parent_id,
                        "qtype": chunk.qtype,
                        "question": chunk.question,
                    },
                )
                for chunk, vector in zip(
                    chunks[start : start + batch_size],
                    vectors[start : start + batch_size],
                    strict=True,
                )
            ]
        )
    return len(chunks)
