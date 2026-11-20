# RAGMed

[![CI](https://github.com/BijanProjects/RAGMed/actions/workflows/ci.yml/badge.svg)](https://github.com/BijanProjects/RAGMed/actions/workflows/ci.yml)

RAGMed is a multi-agent clinical triage assistant. A patient describes their symptoms and gets an
answer grounded in a curated medical knowledge base, with citations. When a visit is warranted, the
system offers real appointment slots and books one after the patient confirms.

> **Status:** active rebuild. See the roadmap below for what exists today.

## How the pieces fit

```text
 frontend/ (Next.js on Vercel)
      │  A2A over HTTPS
      ▼
 backend/  Care Coordinator agent ──A2A──▶ Diagnosis agent ──MCP──▶ Medical Knowledge MCP ──▶ vector store
                 │                └─A2A──▶ Scheduling agent ──MCP──▶ Scheduling MCP ──▶ Cal.com
                 └──▶ Postgres (conversation state) · Valkey (rate limits, cache)
```

- **A2A** (Agent2Agent) is how agents talk to each other: each agent is a separate web service
  that publishes an *Agent Card* describing what it can do.
- **MCP** (Model Context Protocol) is how agents reach tools and data. Agents never call a database
  or an external API directly; they call tools on an MCP server.
- **RAG** (retrieval-augmented generation): the Diagnosis agent first retrieves relevant passages
  from the knowledge base, then answers using only those passages and cites them.

## Project map

| Folder | What lives there | Start reading at |
| --- | --- | --- |
| [`backend/`](backend/) | Everything Python: retrieval, MCP servers, agents, evaluation | [`backend/README.md`](backend/README.md) |
| [`frontend/`](frontend/) | The Next.js web app (not started yet) | [`frontend/README.md`](frontend/README.md) |
| `.github/workflows/` | CI: lint, type checks, tests and secret scanning on every pull request | `ci.yml` |

## Quick start (backend)

```bash
cd backend
make install   # create .venv, install locked dependencies and git hooks
make check     # lint, strict type check, tests with coverage
```

## Roadmap

- [x] Workspace, tooling and CI
- [ ] Ingestion pipeline and embedding benchmark *(in progress)*
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
