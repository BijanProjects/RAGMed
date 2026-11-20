import itertools
import os
import uuid
from collections.abc import AsyncIterator

import pytest

from ragmed_vectorstores.base import VectorStore
from ragmed_vectorstores.contract import VectorStoreContract
from ragmed_vectorstores.memory import InMemoryStore
from ragmed_vectorstores.qdrant import QdrantStore

_names = itertools.count()


class TestInMemoryStore(VectorStoreContract):
    @pytest.fixture
    async def store(self) -> VectorStore:
        return InMemoryStore()


class TestQdrantStore(VectorStoreContract):
    """Qdrant embedded in the test process; no server or container needed."""

    @pytest.fixture
    async def store(self) -> AsyncIterator[VectorStore]:
        store = QdrantStore(f"test-{next(_names)}", location=":memory:")
        yield store
        await store.close()


QDRANT_URL = os.environ.get("RAGMED_TEST_QDRANT_URL")


@pytest.mark.skipif(not QDRANT_URL, reason="set RAGMED_TEST_QDRANT_URL to test a Qdrant server")
class TestQdrantServer(VectorStoreContract):
    """The same contract against a real Qdrant server, e.g. one started with Docker."""

    @pytest.fixture
    async def store(self) -> AsyncIterator[VectorStore]:
        store = QdrantStore(f"contract-{uuid.uuid4().hex[:12]}", url=QDRANT_URL)
        yield store
        await store.drop()
        await store.close()
