"""Sparse (BM25) retrieval, rank fusion and reranking for the hybrid pipeline.

The benchmark uses an in-memory BM25 with English stemming and stop words, which is close
to what Qdrant's BM25 sparse vectors do in production.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

RRF_K = 60


class SparseRetriever(Protocol):
    def search(self, queries: Sequence[str], k: int) -> list[list[int]]:
        """Indices of the best-matching documents per query, best first."""
        ...


class BM25Retriever:
    def __init__(self, documents: Sequence[str]) -> None:
        import bm25s  # noqa: PLC0415 - optional dependency
        import Stemmer  # noqa: PLC0415

        self._bm25s = bm25s
        self._stemmer = Stemmer.Stemmer("english")
        self._index = bm25s.BM25()
        self._index.index(self._tokenize(documents, as_ids=True), show_progress=False)
        self._size = len(documents)

    def _tokenize(self, texts: Sequence[str], *, as_ids: bool) -> Any:
        return self._bm25s.tokenize(
            list(texts),
            stopwords="en",
            stemmer=self._stemmer,
            return_ids=as_ids,
            show_progress=False,
        )

    def search(self, queries: Sequence[str], k: int) -> list[list[int]]:
        k = min(k, self._size)
        docs, scores = self._index.retrieve(
            self._tokenize(queries, as_ids=False), k=k, show_progress=False
        )
        return [
            [int(d) for d, s in zip(row_docs, row_scores, strict=True) if s > 0]
            for row_docs, row_scores in zip(docs, scores, strict=True)
        ]


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        """Relevance score per passage; higher is more relevant."""
        ...


def rerank_head(
    query: str,
    candidates: Sequence[int],
    texts: Sequence[str],
    reranker: Reranker,
    depth: int,
) -> list[int]:
    """Rerank the first `depth` candidates; the rest keep their order after them."""
    head, tail = list(candidates[:depth]), list(candidates[depth:])
    if not head:
        return tail
    scores = reranker.score(query, [texts[i] for i in head])
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return [head[i] for i in order] + tail


class CrossEncoderReranker:  # pragma: no cover - needs model weights
    """Any sentence-transformers CrossEncoder, e.g. BAAI/bge-reranker-v2-m3."""

    def __init__(self, model_id: str, *, batch_size: int = 32, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._name = model_id
        self._batch_size = batch_size
        self._model = CrossEncoder(model_id, max_length=max_length)

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


QWEN3_RERANK_INSTRUCTION = "Given a patient's health question, judge whether the passage answers it"


class Qwen3Reranker:  # pragma: no cover - needs model weights
    """Qwen3-Reranker: a causal LM scored by P("yes") for a query-document judgement.

    Follows the prompt format from the model card; it is not a plain cross-encoder.
    """

    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        'the Query and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n'
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        instruction: str = QWEN3_RERANK_INSTRUCTION,
        batch_size: int = 8,
        max_length: int = 1024,
        device: str | None = None,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._name = model_id
        self._instruction = instruction
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(self._device).eval()
        self._yes = self._tokenizer.convert_tokens_to_ids("yes")
        self._no = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix_ids = self._tokenizer.encode(self._PREFIX, add_special_tokens=False)
        self._suffix_ids = self._tokenizer.encode(self._SUFFIX, add_special_tokens=False)

    @property
    def name(self) -> str:
        return self._name

    def _batch_scores(self, query: str, passages: Sequence[str]) -> list[float]:
        budget = self._max_length - len(self._prefix_ids) - len(self._suffix_ids)
        bodies = [
            f"<Instruct>: {self._instruction}\n<Query>: {query}\n<Document>: {p}" for p in passages
        ]
        encoded = self._tokenizer(
            bodies, truncation="longest_first", max_length=budget,
            add_special_tokens=False, return_attention_mask=False,
        )  # fmt: skip
        ids = [self._prefix_ids + row + self._suffix_ids for row in encoded["input_ids"]]
        batch = self._tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        batch = batch.to(self._device)
        with self._torch.inference_mode():
            logits = self._model(**batch).logits[:, -1, :]
        pair = self._torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1)
        probs = self._torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].exp()
        return [float(p) for p in probs.cpu()]

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        scores: list[float] = []
        for start in range(0, len(passages), self._batch_size):
            scores.extend(self._batch_scores(query, passages[start : start + self._batch_size]))
        return scores


# Smallest first: 23M, 278M, 568M and 600M parameters.
RERANKER_PRESETS: dict[str, Callable[[], Reranker]] = {
    "minilm-ce": lambda: CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L6-v2"),
    "bge-reranker-base": lambda: CrossEncoderReranker("BAAI/bge-reranker-base"),
    "bge-reranker-v2-m3": lambda: CrossEncoderReranker("BAAI/bge-reranker-v2-m3"),
    "qwen3-reranker-0.6b": Qwen3Reranker,
}
