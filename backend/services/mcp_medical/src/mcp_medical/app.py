"""Wire the server from settings: corpus, vector store, embedder, auth."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastmcp import FastMCP
from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict

from mcp_medical.auth import ServiceTokenVerifier
from mcp_medical.knowledge import KnowledgeBase, index_chunks
from mcp_medical.server import DatasetCard, build_server
from ragmed_core.settings import CoreSettings, Environment
from ragmed_retrieval.chunking import chunk_pairs
from ragmed_retrieval.corpus import load_medquad
from ragmed_retrieval.embedding import Embedder
from ragmed_retrieval.onnx import OnnxEmbedder
from ragmed_vectorstores.qdrant import QdrantStore

EmbedderFactory = Callable[["MedicalSettings"], Embedder]


class MedicalSettings(CoreSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGMED_", env_file=".env", extra="ignore", frozen=True
    )

    service_name: str = "mcp-medical"
    host: str = "127.0.0.1"
    port: int = 9001
    corpus: Path = Path("data/raw/medquad.csv")
    onnx_dir: Path = Path("data/artifacts/onnx/BAAI__bge-m3")
    # A Qdrant server URL in production; otherwise an embedded store in this folder.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Path("data/artifacts/qdrant")
    collection: str = "medquad"
    # JSON object of caller name to token, e.g. {"diagnosis-agent": "..."}.
    service_tokens: dict[str, SecretStr] = Field(default_factory=dict)


def _store(settings: MedicalSettings) -> QdrantStore:
    if settings.qdrant_url:
        key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantStore(settings.collection, url=settings.qdrant_url, api_key=key)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantStore(settings.collection, path=str(settings.qdrant_path))


def _index_info_path(settings: MedicalSettings) -> Path:
    return settings.qdrant_path / f"{settings.collection}-index.json"


def _onnx_embedder(settings: MedicalSettings) -> Embedder:  # pragma: no cover - needs the export
    return OnnxEmbedder(settings.onnx_dir)


async def ingest(
    settings: MedicalSettings,
    vectors_file: Path | None,
    *,
    embedder_factory: EmbedderFactory = _onnx_embedder,
) -> int:
    """Load every passage and its vector into the store.

    Pass `vectors_file` (the .npy the benchmark saved, embedded once on a GPU in full
    precision) to skip embedding here; otherwise passages are embedded with the server's model.
    """
    pairs, _ = load_medquad(settings.corpus)
    chunks = chunk_pairs(pairs)
    if vectors_file is not None:
        vectors = np.load(vectors_file)
        model = f"from {vectors_file.name}"
    else:
        embedder = embedder_factory(settings)
        vectors = embedder.embed_documents([c.text for c in chunks])
        model = embedder.name
    store = _store(settings)
    try:
        count = await index_chunks(store, chunks, vectors)
    finally:
        await store.close()
    info = {
        "indexed_on": datetime.now(UTC).date().isoformat(),
        "passages": count,
        "answers": len(pairs),
        "vectors": model,
    }
    _index_info_path(settings).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return count


def create_server(
    settings: MedicalSettings, *, embedder_factory: EmbedderFactory = _onnx_embedder
) -> FastMCP:
    """The configured server. Refuses to start in production without service tokens."""
    tokens = {name: token.get_secret_value() for name, token in settings.service_tokens.items()}
    if not tokens and settings.environment is Environment.PRODUCTION:
        raise RuntimeError("RAGMED_SERVICE_TOKENS must be set in production")

    pairs, _ = load_medquad(settings.corpus)
    embedder = embedder_factory(settings)
    knowledge = KnowledgeBase(embedder, _store(settings), {p.id: p for p in pairs})

    info_path = _index_info_path(settings)
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    card = DatasetCard(
        name="MedQuAD",
        licence="CC BY 4.0",
        citation="Ben Abacha A, Demner-Fushman D. A question-entailment approach to question "
        "answering. BMC Bioinformatics 20, 511 (2019)",
        url="https://github.com/abachaa/MedQuAD",
        answers=len(pairs),
        passages=int(info.get("passages", 0)),
        embedding_model=embedder.name,
        indexed_on=str(info.get("indexed_on", "not indexed yet")),
    )
    return build_server(knowledge, card, auth=ServiceTokenVerifier(tokens) if tokens else None)
