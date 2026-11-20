# ragmed-vectorstores

One async interface over vector databases. RAGMed stores its knowledge base in Qdrant by default,
and users can connect their own database by picking an adapter.

```python
from ragmed_vectorstores import Document, Query, load_store

store = load_store("qdrant", url="https://qdrant.example.com", api_key="...")
await store.upsert([Document("p1", "Asthma causes wheezing.", vector, {"qtype": "symptoms"})])
hits = await store.query(Query(question_vector, top_k=5, filter={"qtype": "symptoms"}))
```

## Adapters

| Name | Backend | Filters | Notes |
| --- | --- | :---: | --- |
| `qdrant` | [Qdrant](https://qdrant.tech), self-hosted or Qdrant Cloud | ✅ | Default. `location=":memory:"` runs it inside Python, no server |
| `memory` | NumPy, in process | ✅ | Tests, local development, and the reference implementation |

Planned: pgvector, Weaviate and Pinecone. Their contract tests need real databases running in
Docker (testcontainers).

## The interface

Every adapter implements [`VectorStore`](src/ragmed_vectorstores/base.py):

| Method | Does |
| --- | --- |
| `capabilities` | What the backend supports: hybrid search, filters, maximum dimensions |
| `health()` | Whether the database answers |
| `describe()` | Collection name, vector size, number of documents |
| `upsert(docs)` | Insert, replacing documents with the same id. The collection is created on the first upsert |
| `query(q)` | Closest documents first, scored by cosine similarity, optionally filtered by exact metadata values |
| `delete(ids)` | Remove documents; unknown ids are ignored |

Rules every adapter follows, enforced by the contract suite:

- Scores are cosine similarity, so a vector and the same vector scaled up score 1.0.
- A batch with a vector of the wrong length raises `DimensionMismatchError` and stores nothing.
- Filters match exactly and accept text, integers and booleans; a float filter raises `TypeError`.
- Metadata comes back exactly as it went in.

## Writing your own adapter

1. Write a class with the methods above. Any class with them counts; no base class is needed.
2. Run the shared contract suite against it. It is the same suite the built-in adapters pass:

   ```python
   import pytest
   from ragmed_vectorstores.contract import VectorStoreContract


   class TestMyStore(VectorStoreContract):
       @pytest.fixture
       async def store(self):
           store = MyStore(collection="test")  # a fresh, empty store
           yield store
           await store.drop()  # clean up
   ```

3. Register it in your package's `pyproject.toml`, so `load_store("my-db", ...)` finds it after a
   plain `pip install`, with no change to RAGMed:

   ```toml
   [project.entry-points."ragmed.vectorstores"]
   my-db = "my_package.store:MyStore"
   ```

The suite needs `pytest` and `anyio`.
