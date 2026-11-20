"""Embedding model interface and a sentence-transformers implementation."""

from collections.abc import Callable, Sequence
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
        # Some checkpoints (e.g. Qwen3) default to bfloat16, which is slow on CPU and would make
        # model comparisons unfair; every model runs in float32.
        self._model = SentenceTransformer(
            model_name, device=device, model_kwargs={"dtype": "float32"}
        )
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


TextEncoder = Callable[[Sequence[str]], npt.NDArray[np.floating[Any]]]


class DualEncoderEmbedder:
    """Asymmetric model with separate query and document encoders (e.g. MedCPT)."""

    def __init__(
        self,
        name: str,
        *,
        encode_queries: TextEncoder,
        encode_documents: TextEncoder,
        batch_size: int = 32,
    ) -> None:
        self._name = name
        self._encode_queries = encode_queries
        self._encode_documents = encode_documents
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._name

    def _batched(self, encode: TextEncoder, texts: Sequence[str]) -> Vectors:
        parts = [
            encode(texts[start : start + self._batch_size])
            for start in range(0, len(texts), self._batch_size)
        ]
        return l2_normalize(np.concatenate(parts)) if parts else np.zeros((0, 0), np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._batched(self._encode_documents, texts)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._batched(self._encode_queries, texts)


MEDCPT = "ncbi/MedCPT"


def _cls_encoder(model_id: str, max_length: int, device: str) -> TextEncoder:  # pragma: no cover
    """Hugging Face encoder returning the [CLS] vector, as MedCPT was trained to use."""
    import torch  # noqa: PLC0415 - heavy, optional
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    def encode(texts: Sequence[str]) -> npt.NDArray[np.floating[Any]]:
        batch = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state[:, 0, :]
        return np.asarray(hidden.float().cpu().numpy())

    return encode


def medcpt_embedder(device: str | None = None) -> DualEncoderEmbedder:  # pragma: no cover
    """MedCPT (NCBI): PubMed-trained query and article encoders."""
    import torch  # noqa: PLC0415

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    return DualEncoderEmbedder(
        MEDCPT,
        encode_queries=_cls_encoder("ncbi/MedCPT-Query-Encoder", 64, device),
        encode_documents=_cls_encoder("ncbi/MedCPT-Article-Encoder", 512, device),
    )
