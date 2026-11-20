import csv
import json
from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client

from mcp_medical.app import MedicalSettings, create_server, ingest
from mcp_medical.auth import ServiceTokenVerifier
from ragmed_core.settings import Environment
from ragmed_retrieval.chunking import chunk_pairs
from ragmed_retrieval.corpus import load_medquad
from ragmed_retrieval.embedding import Embedder
from ragmed_retrieval.testing import HashingEmbedder

pytestmark = pytest.mark.anyio

ROWS = [
    ("symptoms", "What are the symptoms of asthma?", "Asthma causes wheezing."),
    ("treatment", "How is diabetes treated?", "Diabetes is treated with insulin."),
    ("causes", "What causes influenza?", "Influenza is caused by a virus."),
]


@pytest.fixture
def settings(tmp_path: Path) -> MedicalSettings:
    corpus = tmp_path / "medquad.csv"
    with corpus.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qtype", "Question", "Answer"])
        writer.writerows(ROWS)
    return MedicalSettings(corpus=corpus, qdrant_path=tmp_path / "qdrant")


def hashing(_: MedicalSettings) -> Embedder:
    return HashingEmbedder()


async def test_ingest_then_serve_answers_searches(settings: MedicalSettings) -> None:
    assert await ingest(settings, None, embedder_factory=hashing) == len(ROWS)

    async with Client(create_server(settings, embedder_factory=hashing)) as client:
        result = await client.call_tool("search_medical_knowledge", {"query": "insulin diabetes"})
        card = await client.read_resource("ragmed://dataset/card")

    assert result.structured_content is not None
    assert result.structured_content["result"][0]["question"] == "How is diabetes treated?"
    info = json.loads(card[0].text)
    assert info["passages"] == len(ROWS)
    assert info["embedding_model"] == "test/hashing"


async def test_ingest_can_use_precomputed_vectors(
    settings: MedicalSettings, tmp_path: Path
) -> None:
    chunks = chunk_pairs(load_medquad(settings.corpus)[0])
    vectors_file = tmp_path / "vectors.npy"
    np.save(vectors_file, HashingEmbedder().embed_documents([c.text for c in chunks]))

    def must_not_embed(_: MedicalSettings) -> Embedder:
        raise AssertionError("vectors were given, so nothing should be embedded")

    count = await ingest(settings, vectors_file, embedder_factory=must_not_embed)

    assert count == len(chunks)


async def test_ingest_refuses_vectors_that_do_not_match(
    settings: MedicalSettings, tmp_path: Path
) -> None:
    vectors_file = tmp_path / "wrong.npy"
    np.save(vectors_file, np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="vectors"):
        await ingest(settings, vectors_file)


def test_production_requires_service_tokens(settings: MedicalSettings) -> None:
    production = settings.model_copy(update={"environment": Environment.PRODUCTION})

    with pytest.raises(RuntimeError, match="RAGMED_SERVICE_TOKENS"):
        create_server(production, embedder_factory=hashing)


def test_service_tokens_turn_on_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: MedicalSettings
) -> None:
    monkeypatch.setenv("RAGMED_SERVICE_TOKENS", json.dumps({"diagnosis-agent": "t" * 40}))
    configured = MedicalSettings(corpus=settings.corpus, qdrant_path=tmp_path / "q2")

    server = create_server(configured, embedder_factory=hashing)

    assert isinstance(server.auth, ServiceTokenVerifier)
    assert "t" * 40 not in repr(configured)  # the token is a secret, never printed
