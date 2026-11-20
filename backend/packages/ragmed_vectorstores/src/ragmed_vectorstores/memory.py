"""A vector store held in memory: for tests, local development and as a reference adapter."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ragmed_vectorstores.base import DimensionMismatchError
from ragmed_vectorstores.models import (
    Capabilities,
    CollectionInfo,
    Document,
    Health,
    Hit,
    Query,
    UpsertResult,
    check_filter,
)


def unit(vector: Sequence[float]) -> npt.NDArray[np.float32]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array


class InMemoryStore:
    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._docs: dict[str, Document] = {}
        self._vectors: dict[str, npt.NDArray[np.float32]] = {}
        self._dimensions: int | None = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(hybrid=False, filters=True)

    async def health(self) -> Health:
        return Health(ok=True, detail="in memory")

    async def describe(self) -> CollectionInfo:
        return CollectionInfo(self._name, self._dimensions, len(self._docs))

    async def upsert(self, docs: Sequence[Document]) -> UpsertResult:
        if not docs:
            return UpsertResult(0)
        expected = self._dimensions or len(docs[0].vector)
        for doc in docs:  # check everything first, so a bad batch stores nothing
            if len(doc.vector) != expected:
                raise DimensionMismatchError(
                    f"{doc.id}: vector has {len(doc.vector)} dimensions, collection has {expected}"
                )
        self._dimensions = expected
        for doc in docs:
            self._docs[doc.id] = doc
            self._vectors[doc.id] = unit(doc.vector)
        return UpsertResult(len(docs))

    async def query(self, q: Query) -> list[Hit]:
        check_filter(q.filter)
        if self._dimensions is not None and len(q.vector) != self._dimensions:
            raise DimensionMismatchError(
                f"query has {len(q.vector)} dimensions, collection has {self._dimensions}"
            )
        wanted = q.filter or {}
        candidates = [
            doc_id
            for doc_id, doc in self._docs.items()
            if all(doc.metadata.get(key) == value for key, value in wanted.items())
        ]
        if not candidates:
            return []
        scores = np.stack([self._vectors[i] for i in candidates]) @ unit(q.vector)
        order = np.argsort(-scores, kind="stable")[: q.top_k]
        return [
            Hit(
                candidates[i],
                float(scores[i]),
                self._docs[candidates[i]].text,
                self._docs[candidates[i]].metadata,
            )
            for i in order
        ]

    async def delete(self, ids: Sequence[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)
            self._vectors.pop(doc_id, None)
