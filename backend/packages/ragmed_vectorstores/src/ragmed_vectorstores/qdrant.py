"""Qdrant adapter: the default store, self-hosted on the server or Qdrant Cloud.

Three ways to run it:

- `QdrantStore(url=..., api_key=...)`: a Qdrant server (production).
- `QdrantStore(path="data/qdrant")`: embedded, saved to a folder; local development without
  Docker. Only one process can open the folder at a time.
- `QdrantStore(location=":memory:")`: embedded and temporary; tests.
"""

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from ragmed_vectorstores.base import DimensionMismatchError
from ragmed_vectorstores.models import (
    Capabilities,
    CollectionInfo,
    Document,
    Health,
    Hit,
    Metadata,
    Query,
    UpsertResult,
    check_filter,
)

# Qdrant point ids must be integers or UUIDs. Our ids are strings, so each is turned into a
# stable UUID and the original id is kept in the payload with the text.
_ID_NAMESPACE = uuid.UUID("5b3f6c0e-7d0a-4f8e-9c1a-2e6d4b8a9f10")
ID_KEY, TEXT_KEY = "_id", "_text"


def point_id(doc_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, doc_id))


class QdrantStore:
    def __init__(
        self,
        collection: str = "ragmed",
        *,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        path: str | None = None,
        timeout: int = 10,
        check_compatibility: bool = True,
    ) -> None:
        self._collection = collection
        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            location=location,
            path=path,
            timeout=timeout,
            # Compares client and server versions in a background thread; off in some tests.
            check_compatibility=check_compatibility,
        )

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(hybrid=True, filters=True, max_dimensions=65_536)

    async def health(self) -> Health:
        try:
            await self._client.get_collections()
        except Exception as error:
            return Health(ok=False, detail=f"{type(error).__name__}: {error}")
        return Health(ok=True)

    async def _dimensions(self) -> int | None:
        if not await self._client.collection_exists(self._collection):
            return None
        info = await self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise TypeError(f"{self._collection} uses named vectors, which are not supported")
        return vectors.size

    async def describe(self) -> CollectionInfo:
        dimensions = await self._dimensions()
        if dimensions is None:
            return CollectionInfo(self._collection, None, 0)
        count = await self._client.count(self._collection, exact=True)
        return CollectionInfo(self._collection, dimensions, count.count)

    async def upsert(self, docs: Sequence[Document]) -> UpsertResult:
        if not docs:
            return UpsertResult(0)
        for doc in docs:
            if ID_KEY in doc.metadata or TEXT_KEY in doc.metadata:
                raise ValueError(f"{doc.id}: metadata keys {ID_KEY} and {TEXT_KEY} are reserved")
        dimensions = await self._dimensions()
        if dimensions is None:
            dimensions = len(docs[0].vector)
            await self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE
                ),
            )
        for doc in docs:
            if len(doc.vector) != dimensions:
                raise DimensionMismatchError(
                    f"{doc.id}: vector has {len(doc.vector)} dimensions, collection has "
                    f"{dimensions}"
                )
        await self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=point_id(doc.id),
                    vector=list(doc.vector),
                    payload={ID_KEY: doc.id, TEXT_KEY: doc.text, **doc.metadata},
                )
                for doc in docs
            ],
        )
        return UpsertResult(len(docs))

    async def query(self, q: Query) -> list[Hit]:
        check_filter(q.filter)
        dimensions = await self._dimensions()
        if dimensions is None:
            return []
        if len(q.vector) != dimensions:
            raise DimensionMismatchError(
                f"query has {len(q.vector)} dimensions, collection has {dimensions}"
            )
        conditions: list[models.Condition] = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in (q.filter or {}).items()
        ]
        response = await self._client.query_points(
            self._collection,
            query=list(q.vector),
            limit=q.top_k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            with_payload=True,
        )
        return [_hit(point.score, point.payload or {}) for point in response.points]

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids or not await self._client.collection_exists(self._collection):
            return
        await self._client.delete(
            self._collection,
            points_selector=models.PointIdsList(points=[point_id(i) for i in ids]),
        )

    async def drop(self) -> None:
        """Delete the whole collection, e.g. before re-indexing from scratch."""
        if await self._client.collection_exists(self._collection):
            await self._client.delete_collection(self._collection)

    async def close(self) -> None:
        await self._client.close()


def _hit(score: float, payload: dict[str, Any]) -> Hit:
    metadata: Metadata = {k: v for k, v in payload.items() if k not in (ID_KEY, TEXT_KEY)}
    return Hit(str(payload[ID_KEY]), float(score), str(payload[TEXT_KEY]), metadata)
