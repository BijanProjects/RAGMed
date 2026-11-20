"""Embedding model interface and a sentence-transformers implementation."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]

_PROGRESS_THRESHOLD = 1000


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_queries(self, texts: Sequence[str]) -> Vectors: ...


def l2_normalize(vectors: npt.NDArray[np.floating[Any]]) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized: Vectors = (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)
    return normalized


class SentenceTransformerEmbedder:
    """Any sentence-transformers model, with optional instruction prompts for queries.

    Instruction-tuned models such as Qwen3-Embedding expect queries to carry a task
    prompt while documents are embedded as-is.
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_prompt_name: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int | None = 512,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy, optional

        self._model_name = model_name
        self._query_prompt_name = query_prompt_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length

    @property
    def name(self) -> str:
        return self._model_name

    def _encode(self, texts: Sequence[str], prompt_name: str | None) -> Vectors:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            prompt_name=prompt_name,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > _PROGRESS_THRESHOLD,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode(texts, None)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._encode(texts, self._query_prompt_name)
