"""Lightweight fakes for tests and offline demos (no model download needed)."""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from ragmed_retrieval.embedding import Vectors, l2_normalize

_WORD = re.compile(r"[a-z]+")


class HashingEmbedder:
    """Bag-of-words embedder: each word is hashed into one of `dimensions` buckets.

    It has no semantic understanding, but texts that share words get similar vectors,
    which is enough to test ranking code deterministically.
    """

    name = "test/hashing"

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions
        self.document_calls = 0

    def _embed(self, texts: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                bucket = int(hashlib.blake2b(word.encode(), digest_size=8).hexdigest(), 16)
                vectors[row, bucket % self.dimensions] += 1
        return l2_normalize(vectors)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.document_calls += 1
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._embed(texts)
