# RAGMed

[![CI](https://github.com/BijanProjects/RAGMed/actions/workflows/ci.yml/badge.svg)](https://github.com/BijanProjects/RAGMed/actions/workflows/ci.yml)

RAGMed is a multi-agent clinical triage assistant. A patient describes their symptoms and gets an
answer grounded in a curated medical knowledge base, with citations. When a visit is warranted, the
system offers real appointment slots and books one after the patient confirms.

> **Status:** active rebuild. The original single-script FAISS prototype lives in
> [`legacy/`](legacy/). The production system is being built incrementally; see the roadmap below.

## Architecture

```text
Next.js (Vercel)  ──A2A──▶  Care Coordinator ──A2A──▶ Diagnosis agent ──MCP──▶ Medical Knowledge MCP ──▶ Qdrant
                                     │        ──A2A──▶ Scheduling agent ──MCP──▶ Scheduling MCP ──▶ Cal.com
                                     └──▶ Postgres (LangGraph checkpoints, A2A tasks) · Valkey (rate limits)
```

- **Agents:** LangGraph graphs, each served as its own A2A service with an Agent Card.
- **Tools and data:** two MCP servers. Agents never touch a database or external API directly.
- **Retrieval:** hybrid dense + BM25 search with a cross-encoder reranker, behind a pluggable
  vector store interface (Qdrant, Pinecone, Weaviate, pgvector, FAISS).
- **Models:** open-weight models served by Groq, with Gemini as fallback.
- **Observability:** LangSmith for development and evaluation, Langfuse for production traces.

## Repository layout

| Path | Contents |
| --- | --- |
| `packages/ragmed_core` | Settings, structured logging with PII redaction, W3C trace context |
| `data/raw` | MedQuAD question–answer corpus |
| `legacy/` | Original prototype, kept for reference until fully ported |

## Development

Requires Python 3.12.

```bash
make install   # create .venv, install locked dependencies and git hooks
make check     # lint, strict type check, tests with coverage
```

Run `make help` for every target. Configuration is read from environment variables; copy
`.env.example` to `.env` for local overrides.

## Roadmap

- [x] Workspace, tooling and CI
- [ ] Ingestion pipeline and embedding benchmark
- [ ] Vector store connectors with shared contract tests
- [ ] Medical Knowledge MCP server
- [ ] Scheduling MCP server
- [ ] Diagnosis, Scheduling and Care Coordinator agents over A2A
- [ ] Observability and evaluation suite
- [ ] Web frontend
- [ ] Infrastructure as code and continuous deployment

## Data

[MedQuAD](https://github.com/abachaa/MedQuAD) (Ben Abacha & Demner-Fushman, 2019), distributed
via the [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset)
on Kaggle, licensed CC BY 4.0.

## Disclaimer

RAGMed is an educational project. It is not a medical device and does not provide diagnoses. In an
emergency, contact your local emergency services.
