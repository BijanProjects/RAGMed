"""Run an embedding model with ONNX Runtime on CPU, without PyTorch.

`export_onnx` (offline, needs PyTorch) turns a Hugging Face encoder into one ONNX graph that
already includes pooling and normalisation, then quantizes its weights to int8. `OnnxEmbedder`
runs that graph with only `onnxruntime` and `tokenizers`, which is what the server will use.

Measured for BGE-M3 on 4 CPU threads: 10 ms per question instead of 58 ms with PyTorch, a
569 MB model instead of 2.3 GB, and no loss of retrieval accuracy (see the README).
"""

import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ragmed_retrieval.embedding import Vectors, l2_normalize

Pooling = Literal["cls", "mean"]
MODEL_FILE = "model_int8.onnx"
TOKENIZER_FILE = "tokenizer.json"
METADATA_FILE = "ragmed_onnx.json"


@dataclass(frozen=True, slots=True)
class OnnxMetadata:
    model_id: str
    pooling: Pooling
    max_length: int
    pad_id: int
    pad_token: str
    query_prefix: str = ""

    @classmethod
    def load(cls, model_dir: Path) -> "OnnxMetadata":
        return cls(**json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8")))

    def save(self, model_dir: Path) -> None:
        text = json.dumps(asdict(self), indent=2) + "\n"
        (model_dir / METADATA_FILE).write_text(text, encoding="utf-8")


class OnnxEmbedder:
    """Embeds text with an exported ONNX model.

    Its `name` is the original model id: the int8 graph produces vectors in the same space as
    the original model, so passages embedded once with the full model can be searched with it.
    """

    def __init__(self, model_dir: Path, *, threads: int = 4, batch_size: int = 32) -> None:
        import onnxruntime as ort  # noqa: PLC0415 - optional runtime dependency
        from tokenizers import Tokenizer  # noqa: PLC0415

        self.metadata = OnnxMetadata.load(model_dir)
        self._batch_size = batch_size
        self._tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
        self._tokenizer.enable_truncation(max_length=self.metadata.max_length)
        self._tokenizer.enable_padding(
            pad_id=self.metadata.pad_id, pad_token=self.metadata.pad_token
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_dir / MODEL_FILE), options, providers=["CPUExecutionProvider"]
        )

    @property
    def name(self) -> str:
        return self.metadata.model_id

    def _encode_batch(self, texts: Sequence[str]) -> Vectors:
        encodings = self._tokenizer.encode_batch(list(texts))
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        (vectors,) = self._session.run(["embedding"], inputs)
        return l2_normalize(np.asarray(vectors))

    def _encode(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(
            [
                self._encode_batch(texts[start : start + self._batch_size])
                for start in range(0, len(texts), self._batch_size)
            ]
        )

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        prefix = self.metadata.query_prefix
        return self._encode([prefix + t for t in texts] if prefix else texts)


def export_onnx(  # pragma: no cover - needs PyTorch and model weights
    model_id: str,
    out_dir: Path,
    *,
    pooling: Pooling,
    max_length: int = 512,
    query_prefix: str = "",
) -> Path:
    """Export `model_id` with pooling and normalisation built in, then quantize it to int8."""
    import torch  # noqa: PLC0415
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    class PooledEncoder(torch.nn.Module):  # type: ignore[misc]  # torch is untyped for mypy here
        def __init__(self, encoder: Any) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = hidden.last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=-1)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = PooledEncoder(AutoModel.from_pretrained(model_id, dtype=torch.float32).eval())
    example = tokenizer(["example question"], return_tensors="pt")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Full-precision graph is only an intermediate: over 2 GB for BGE-M3, so its weights
        # go in an external file, and only the int8 graph is kept.
        full = Path(tmp) / "model.onnx"
        torch.onnx.export(
            model,
            (example["input_ids"], example["attention_mask"]),
            str(full),
            input_names=["input_ids", "attention_mask"],
            output_names=["embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "tokens"},
                "attention_mask": {0: "batch", 1: "tokens"},
                "embedding": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
            external_data=True,
        )
        quantize_dynamic(str(full), str(out_dir / MODEL_FILE), weight_type=QuantType.QInt8)

    tokenizer.backend_tokenizer.save(str(out_dir / TOKENIZER_FILE))
    OnnxMetadata(
        model_id=model_id,
        pooling=pooling,
        max_length=max_length,
        pad_id=int(tokenizer.pad_token_id),
        pad_token=str(tokenizer.pad_token),
        query_prefix=query_prefix,
    ).save(out_dir)
    return out_dir / MODEL_FILE
