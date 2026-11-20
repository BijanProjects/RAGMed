"""The interface every vector store adapter implements."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragmed_vectorstores.models import (
    Capabilities,
    CollectionInfo,
    Document,
    Health,
    Hit,
    Query,
    UpsertResult,
)


class DimensionMismatchError(ValueError):
    """A vector's length differs from the collection's."""


@runtime_checkable
class VectorStore(Protocol):
    """An async vector database holding one collection, scored by cosine similarity.

    Every adapter must pass the shared contract suite in `ragmed_vectorstores.contract`.
    The collection is created on the first upsert, sized to that document's vector.
    """

    @property
    def capabilities(self) -> Capabilities: ...

    async def health(self) -> Health: ...

    async def describe(self) -> CollectionInfo: ...

    async def upsert(self, docs: Sequence[Document]) -> UpsertResult:
        """Insert documents, replacing any with the same id."""
        ...

    async def query(self, q: Query) -> list[Hit]:
        """Closest documents first; `score` is cosine similarity."""
        ...

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove documents; unknown ids are ignored."""
        ...
