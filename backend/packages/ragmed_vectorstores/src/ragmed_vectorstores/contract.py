"""The behaviour every adapter must share, as a pytest suite.

To test an adapter, subclass `VectorStoreContract` and provide a `store` fixture that yields a
fresh, empty store:

    class TestMyStore(VectorStoreContract):
        @pytest.fixture
        async def store(self) -> VectorStore:
            return MyStore(...)

Needs pytest and anyio (test-only dependencies).
"""

from collections.abc import AsyncIterator

import pytest

from ragmed_vectorstores.base import DimensionMismatchError, VectorStore
from ragmed_vectorstores.models import Document, Query

DOCS = [
    Document("asthma", "Asthma causes wheezing.", [1.0, 0.0, 0.0], {"qtype": "symptoms"}),
    Document("wheeze", "Wheezing is a whistling sound.", [0.9, 0.1, 0.0], {"qtype": "symptoms"}),
    Document("diabetes", "Diabetes needs insulin.", [0.0, 1.0, 0.0], {"qtype": "treatment"}),
    Document(
        "insulin", "Insulin lowers blood sugar.", [0.0, 0.9, 0.1],
        {"qtype": "treatment", "rank": 2},
    ),
    Document(
        "flu", "Influenza is caused by a virus.", [0.0, 0.0, 1.0],
        {"qtype": "causes", "rare": False},
    ),
]  # fmt: skip


@pytest.mark.anyio
class VectorStoreContract:
    @pytest.fixture
    def anyio_backend(self) -> str:
        return "asyncio"

    @pytest.fixture
    async def store(self) -> VectorStore | AsyncIterator[VectorStore]:
        """Override: return a fresh, empty store, or yield one and clean up afterwards."""
        raise NotImplementedError("override the store fixture in your test class")

    @pytest.fixture
    async def filled(self, store: VectorStore) -> VectorStore:
        await store.upsert(DOCS)
        return store

    async def test_implements_the_interface(self, store: VectorStore) -> None:
        assert isinstance(store, VectorStore)

    async def test_is_healthy(self, store: VectorStore) -> None:
        assert (await store.health()).ok

    async def test_starts_empty(self, store: VectorStore) -> None:
        info = await store.describe()

        assert info.count == 0
        assert info.dimensions is None
        assert await store.query(Query([1.0, 0.0, 0.0])) == []

    async def test_describe_after_upsert(self, store: VectorStore) -> None:
        result = await store.upsert(DOCS)
        info = await store.describe()

        assert result.count == len(DOCS)
        assert info.count == len(DOCS)
        assert info.dimensions == 3
        assert info.metric == "cosine"

    async def test_empty_upsert_does_nothing(self, store: VectorStore) -> None:
        assert (await store.upsert([])).count == 0
        assert (await store.describe()).count == 0

    async def test_closest_first_with_cosine_scores(self, filled: VectorStore) -> None:
        hits = await filled.query(Query([2.0, 0.0, 0.0], top_k=2))

        assert [h.id for h in hits] == ["asthma", "wheeze"]
        assert hits[0].score == pytest.approx(1.0, abs=1e-4)  # same direction, any length
        assert hits[0].score >= hits[1].score
        assert hits[0].text == "Asthma causes wheezing."

    async def test_top_k_limits_results(self, filled: VectorStore) -> None:
        assert len(await filled.query(Query([1.0, 1.0, 1.0], top_k=3))) == 3
        assert len(await filled.query(Query([1.0, 1.0, 1.0], top_k=50))) == len(DOCS)

    async def test_upsert_replaces_a_document_with_the_same_id(self, filled: VectorStore) -> None:
        await filled.upsert([Document("asthma", "Updated.", [0.0, 0.0, 1.0], {"qtype": "x"})])

        hits = await filled.query(Query([0.0, 0.0, 1.0], top_k=1, filter={"qtype": "x"}))
        assert (await filled.describe()).count == len(DOCS)
        assert [(h.id, h.text) for h in hits] == [("asthma", "Updated.")]

    async def test_filter_keeps_only_matching_metadata(self, filled: VectorStore) -> None:
        if not filled.capabilities.filters:
            pytest.skip("adapter does not support filters")

        hits = await filled.query(Query([1.0, 0.0, 0.0], top_k=5, filter={"qtype": "treatment"}))
        both = await filled.query(Query([1.0, 0.0, 0.0], filter={"qtype": "treatment", "rank": 2}))
        none = await filled.query(Query([1.0, 0.0, 0.0], filter={"qtype": "nothing"}))

        assert {h.id for h in hits} == {"diabetes", "insulin"}
        assert [h.id for h in both] == ["insulin"]
        assert none == []

    async def test_filter_on_a_boolean(self, filled: VectorStore) -> None:
        hits = await filled.query(Query([1.0, 0.0, 0.0], filter={"rare": False}))

        assert [h.id for h in hits] == ["flu"]

    async def test_float_filters_are_rejected(self, filled: VectorStore) -> None:
        with pytest.raises(TypeError, match="float"):
            # Wrong on purpose: type checkers reject this, the store must too at runtime.
            await filled.query(Query([1.0, 0.0, 0.0], filter={"rank": 2.0}))  # type: ignore[dict-item]

    async def test_metadata_round_trips(self, filled: VectorStore) -> None:
        (hit,) = await filled.query(Query([0.0, 0.9, 0.1], top_k=1))

        assert hit.id == "insulin"
        assert dict(hit.metadata) == {"qtype": "treatment", "rank": 2}

    async def test_delete_removes_and_ignores_unknown_ids(self, filled: VectorStore) -> None:
        await filled.delete(["asthma", "no-such-id"])

        hits = await filled.query(Query([1.0, 0.0, 0.0], top_k=10))
        assert "asthma" not in {h.id for h in hits}
        assert (await filled.describe()).count == len(DOCS) - 1

    async def test_delete_on_an_empty_store(self, store: VectorStore) -> None:
        await store.delete(["anything"])

        assert (await store.describe()).count == 0

    async def test_wrong_dimensions_store_nothing(self, filled: VectorStore) -> None:
        bad_batch = [Document("ok", "fine", [1.0, 0.0, 0.0]), Document("bad", "x", [1.0, 0.0])]

        with pytest.raises(DimensionMismatchError):
            await filled.upsert(bad_batch)
        assert (await filled.describe()).count == len(DOCS)

    async def test_query_with_wrong_dimensions_fails(self, filled: VectorStore) -> None:
        with pytest.raises(DimensionMismatchError):
            await filled.query(Query([1.0, 0.0]))
