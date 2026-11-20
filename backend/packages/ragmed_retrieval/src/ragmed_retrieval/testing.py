"""Lightweight fakes for tests and offline demos (no model download needed)."""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

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


TINY_VOCAB = {
    "[PAD]": 0, "[UNK]": 1, "query": 2, "asthma": 3, "wheezing": 4,
    "diabetes": 5, "insulin": 6, "influenza": 7, "virus": 8,
}  # fmt: skip
TINY_DIMENSIONS = len(TINY_VOCAB)  # one direction per word


def write_tiny_onnx_model(model_dir: Path, model_id: str, *, query_prefix: str = "") -> None:
    """Write a tiny ONNX "encoder" with the same inputs and output as a real export.

    Each token id looks up a fixed vector and the output is the first token's vector, as with
    CLS pooling. Needs the `onnx` and `tokenizers` packages (dev dependencies).
    """
    import numpy as np  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers.models import WordLevel  # noqa: PLC0415
    from tokenizers.normalizers import Lowercase  # noqa: PLC0415
    from tokenizers.pre_tokenizers import Whitespace  # noqa: PLC0415

    from ragmed_retrieval.onnx import MODEL_FILE, TOKENIZER_FILE, OnnxMetadata  # noqa: PLC0415

    model_dir.mkdir(parents=True, exist_ok=True)
    rows = len(TINY_VOCAB)
    table = np.eye(rows, TINY_DIMENSIONS, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["table", "input_ids"], ["tokens"], axis=0),
            helper.make_node("Gather", ["tokens", "first"], ["embedding"], axis=1),
        ],
        "tiny",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"]),
        ],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["b", TINY_DIMENSIONS])],
        initializer=[
            helper.make_tensor("table", TensorProto.FLOAT, table.shape, table.flatten()),
            helper.make_tensor("first", TensorProto.INT64, [], [0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(model_dir / MODEL_FILE))

    tokenizer = Tokenizer(WordLevel(TINY_VOCAB, unk_token="[UNK]"))  # noqa: S106 - not a secret
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / TOKENIZER_FILE))
    OnnxMetadata(model_id, "cls", 8, 0, "[PAD]", query_prefix=query_prefix).save(model_dir)
