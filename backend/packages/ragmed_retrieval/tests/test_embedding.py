import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from ragmed_retrieval.embedding import (
    DualEncoderEmbedder,
    SentenceTransformerEmbedder,
    l2_normalize,
)


class FakeSentenceTransformer:
    instances: ClassVar[list["FakeSentenceTransformer"]] = []

    def __init__(
        self, model_name: str, device: str | None = None, model_kwargs: dict[str, Any] | None = None
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs
        self.max_seq_length = 8192
        self.calls: list[dict[str, Any]] = []
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append({"texts": texts, **kwargs})
        return np.ones((len(texts), 4), dtype=np.float64)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    FakeSentenceTransformer.instances.clear()
    return FakeSentenceTransformer


def test_l2_normalize_handles_zero_vectors() -> None:
    vectors = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))

    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors[0], [0.6, 0.8])
    np.testing.assert_allclose(vectors[1], [0.0, 0.0])


def test_queries_use_the_prompt_and_documents_do_not(
    fake_backend: type[FakeSentenceTransformer],
) -> None:
    embedder = SentenceTransformerEmbedder(
        "org/model", query_prompt_name="query", batch_size=8, device="cpu", max_seq_length=256
    )

    docs = embedder.embed_documents(["a", "b"])
    queries = embedder.embed_queries(["q"])

    model = fake_backend.instances[0]
    assert embedder.name == "org/model"
    assert model.device == "cpu"
    assert model.model_kwargs == {"dtype": "float32"}
    assert model.max_seq_length == 256
    assert model.calls[0]["prompt_name"] is None
    assert model.calls[1]["prompt_name"] == "query"
    assert all(call["normalize_embeddings"] and call["batch_size"] == 8 for call in model.calls)
    assert docs.shape == (2, 4)
    assert queries.dtype == np.float32


def test_dual_encoder_routes_texts_to_each_encoder_in_batches() -> None:
    seen: dict[str, list[list[str]]] = {"query": [], "doc": []}

    def encoder(kind: str, width: int) -> Any:
        def encode(texts: Any) -> np.ndarray:
            seen[kind].append(list(texts))
            return np.full((len(texts), width), 2.0)

        return encode

    embedder = DualEncoderEmbedder(
        "org/dual", encode_queries=encoder("query", 3), encode_documents=encoder("doc", 3),
        batch_size=2,
    )  # fmt: skip

    docs = embedder.embed_documents(["a", "b", "c"])
    queries = embedder.embed_queries(["q"])

    assert embedder.name == "org/dual"
    assert seen == {"query": [["q"]], "doc": [["a", "b"], ["c"]]}
    assert docs.shape == (3, 3)
    assert queries.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-6)


def test_dual_encoder_with_no_texts_returns_empty() -> None:
    embedder = DualEncoderEmbedder(
        "x", encode_queries=lambda t: np.zeros((0, 1)), encode_documents=lambda t: np.zeros((0, 1))
    )

    assert embedder.embed_documents([]).shape[0] == 0
