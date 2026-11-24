import pytest

from mcp_medical.knowledge import KnowledgeBase, index_chunks
from mcp_medical.server import DatasetCard
from ragmed_retrieval.chunking import chunk_pairs
from ragmed_retrieval.corpus import QAPair
from ragmed_retrieval.testing import HashingEmbedder
from ragmed_vectorstores.memory import InMemoryStore

PAIRS = [
    QAPair("a1", "symptoms", "What are the symptoms of asthma?", "Asthma causes wheezing."),
    QAPair("a2", "treatment", "How is asthma treated?", "Asthma is treated with inhalers."),
    QAPair("d1", "treatment", "How is diabetes treated?", "Diabetes is treated with insulin."),
    QAPair("f1", "causes", "What causes influenza?", "Influenza is caused by a virus."),
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def card() -> DatasetCard:
    return DatasetCard(
        name="MedQuAD (test)",
        licence="CC BY 4.0",
        citation="Ben Abacha & Demner-Fushman, 2019",
        url="https://github.com/abachaa/MedQuAD",
        answers=len(PAIRS),
        passages=len(PAIRS),
        embedding_model="test/hashing",
        indexed_on="2026-11-20",
    )


@pytest.fixture
async def knowledge() -> KnowledgeBase:
    embedder = HashingEmbedder()
    chunks = chunk_pairs(PAIRS)
    store = InMemoryStore()
    await index_chunks(store, chunks, embedder.embed_documents([c.text for c in chunks]))
    urls = {"d1": "https://medlineplus.gov/diabetes.html"}
    return KnowledgeBase(embedder, store, {p.id: p for p in PAIRS}, urls)
