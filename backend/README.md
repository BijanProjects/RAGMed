# RAGMed backend

All Python code: data ingestion, retrieval, evaluation, and (soon) the MCP servers and agents.

## Layout

```text
backend/
├─ packages/                  reusable libraries, imported by the services
│  ├─ ragmed_core/            settings, logging, trace ids, URL guard, secret encryption
│  ├─ ragmed_retrieval/       corpus loading, chunking, embeddings, benchmark, ONNX runtime
│  ├─ ragmed_vectorstores/    one interface over vector databases (Qdrant, in-memory)
│  ├─ ragmed_a2a/             serving and calling agents over A2A (shared by every agent)
│  └─ ragmed_llm/             typed model calls with fallbacks and rate-limit handling
├─ services/                  deployable programs: MCP servers and agents
│  ├─ mcp_medical/            Medical Knowledge MCP server (search tools over the knowledge base)
│  ├─ diagnosis_agent/        answers medical questions with citations, served over A2A
│  └─ coordinator_agent/      the conversation the web app talks to
├─ data/raw/                  the MedQuAD corpus (input, never modified)
├─ data/artifacts/            generated files such as cached embeddings (git-ignored)
├─ evals/                     the retrieval benchmark
│  ├─ data/                   the frozen set of 1,000 test questions
│  ├─ results/                one JSON file per model and pipeline (+ per-query/ scores)
│  ├─ analysis/               confidence intervals and per-question-type scores
│  ├─ analysis.py             recomputes analysis/ from results/per-query/
│  └─ run_all.sh              runs every experiment: make benchmark-all
├─ legacy/                    the original prototype, kept for reference
├─ requirements/              dev.in (what we want) → dev.txt (exact pinned versions)
├─ pyproject.toml             lint, type-check and test configuration
└─ Makefile                   every common command
```

**Packages vs services.** A *package* is a library: it has no `main` and does nothing on its own.
A *service* is a program you run (a web server, an agent). Services import packages. This keeps
logic testable without starting servers.

## Packages

### `ragmed_core`

| Module | Purpose |
| --- | --- |
| `settings.py` | Reads configuration from `RAGMED_*` environment variables into a typed, validated object |
| `logging.py` | JSON logs with service name, trace ids and automatic masking of emails and phone numbers |
| `tracing.py` | Creates and forwards W3C `traceparent` ids so one user request can be followed across services |
| `urlguard.py` | Checks a user-supplied URL is HTTPS and resolves only to public addresses, blocking server-side request forgery (localhost, private networks, cloud metadata, IPv6 tricks) |
| `secrets.py` | Envelope encryption for users' credentials: a master key wraps one key per tenant, which encrypts that tenant's secrets with AES-256-GCM |

### `ragmed_retrieval`

Read in this order; each module builds on the previous one.

| Module | Purpose |
| --- | --- |
| `corpus.py` | Loads MedQuAD, cleans text, removes duplicates, gives each answer a stable id |
| `chunking.py` | Splits long answers into passages of at most 384 words, at sentence boundaries |
| `embedding.py` | Turns text into vectors; one interface, any sentence-transformers model behind it, plus MedCPT's separate query and article encoders |
| `onnx.py` | Exports a model to int8 ONNX and runs it on CPU without PyTorch: how the server embeds questions (10 ms instead of 58 ms) |
| `metrics.py` | Scores a ranking: Recall@k, nDCG@k, MRR |
| `evalset.py` | Builds the fixed set of 1,000 test questions with their correct answers |
| `hybrid.py` | BM25 keyword search, Reciprocal Rank Fusion, and rerankers (MiniLM cross-encoder, bge-reranker, Qwen3-Reranker) |
| `benchmark.py` | Embeds the corpus, searches for every test question (dense, hybrid, or reranked), and scores the results |
| `cli.py` | Command-line entry point for the steps above |
| `testing.py` | A tiny fake embedder and a tiny ONNX model, so tests run without downloading models |

### `ragmed_vectorstores`

One async interface (`VectorStore`) over vector databases, so users can bring their own. Every
adapter must pass the same contract test suite. See its [README](packages/ragmed_vectorstores/README.md).

| Module | Purpose |
| --- | --- |
| `base.py`, `models.py` | The interface and the data it uses: `Document`, `Query`, `Hit` |
| `memory.py` | In-memory adapter: tests, local development, reference behaviour |
| `qdrant.py` | Qdrant adapter, the default; also runs embedded in Python for tests |
| `registry.py` | `load_store("qdrant", ...)` finds adapters, including third-party ones, by entry point |
| `contract.py` | The shared test suite every adapter must pass |

### `ragmed_a2a`

How every agent is served and how agents call each other, so each agent only writes its logic.

| Module | Purpose |
| --- | --- |
| `server.py` | Builds an agent's Agent Card, JSON-RPC endpoint, bearer-token check and `/healthz`; `start_task` opens a task correctly |
| `client.py` | `RemoteAgent.stream(text)`: calls another agent and yields its progress, then one `Reply` with text and data; forwards the trace id |

## Services

### `mcp_medical`

The Medical Knowledge MCP server: three read-only tools (`search_medical_knowledge`,
`get_passage`, `list_topics`), two resources and a grounded-answer prompt, behind bearer-token
auth. See its [README](services/mcp_medical/README.md).

| Module | Purpose |
| --- | --- |
| `knowledge.py` | Embeds the question, queries the vector store, keeps one passage per answer; loads passages into the store |
| `server.py` | The MCP tools, resources, prompt and `/healthz` |
| `auth.py` | Service tokens, compared in constant time |
| `app.py` | Builds the server from `RAGMED_*` settings; the `ingest` step |

### `diagnosis_agent`

Answers a medical question from the knowledge base, with citations and triage (urgency, whether
to see a clinician, which specialty). See its [README](services/diagnosis_agent/README.md).

| Module | Purpose |
| --- | --- |
| `graph.py` | The LangGraph steps: rewrite the question, search, answer if the evidence is enough (else search once more), check the answer; plus the rules enforced in code |
| `reasoner.py` | The language-model calls, with fallbacks between Groq models and rate-limit handling |
| `knowledge.py` | Talks to the MCP server: search and the versioned answer prompt |
| `executor.py` | Serves the graph over A2A, streaming progress |
| `schema.py`, `app.py` | Data shapes; settings and wiring |

### `coordinator_agent`

The Care Coordinator: the only agent the web app talks to (A2A on port 8000).

| Module | Purpose |
| --- | --- |
| `emergency.py` | Red-flag rules checked before any model call: chest pain, stroke signs, breathing, anaphylaxis, bleeding, unconsciousness, poisoning, sudden severe headache, self-harm (988 crisis line) |
| `router.py` | One small-model call: what the person wants, and their question restated to stand on its own |
| `graph.py` | Screen → route → ask the Diagnosis agent or reply directly → remember the conversation |
| `reply.py` | Builds the reply text (answer, sources, next step, disclaimer) and its data |
| `executor.py`, `app.py` | Serves it over A2A; settings and wiring |

### `ragmed_llm`

`ask_structured(models, Schema, system, user)`: asks the first model for a typed answer and moves
to the next one when it is rate limited or returns something unparseable. Groq's free tier allows
8,000 tokens a minute per model, so this is what keeps the demo answering.

## Commands

```bash
make help          # list everything
make install       # set up .venv and git hooks
make install-bench # add PyTorch and the model libraries (needed for search and benchmarks)
make check         # what CI runs: ruff, mypy --strict, pytest with coverage
make lock          # re-pin dependencies after editing requirements/*.in

make export-onnx   # build the fast int8 model the server uses (once, needs install-bench)
make search Q="What are the symptoms of asthma?"   # ask the knowledge base (BGE-M3, CPU)
make report        # print the benchmark results table
make significance  # recompute the confidence intervals
make benchmark-all # re-run every experiment (~1 h on a GPU); tail -f data/artifacts/benchmark.log
make mcp-ingest    # load the knowledge base into an embedded Qdrant (~10 s)
make mcp-serve     # run the MCP server on http://127.0.0.1:9001/mcp
```

The first `make search` embeds all 19,020 passages and saves the vectors to `data/artifacts/`.
This takes about 10 minutes on an Apple Silicon GPU, or much longer on CPU only. Every later
search loads the saved vectors and only embeds your question, which takes about 60 ms.

Retrieval CLI (run from `backend/` with the venv active):

```bash
# Ask questions; defaults: BGE-M3, CPU, 4 threads (like the Oracle server), top 5
python -m ragmed_retrieval search "how is type 2 diabetes treated" "rash and high fever"

# Score one model on the evaluation set; --hybrid adds BM25, --rerank adds a reranker
python -m ragmed_retrieval benchmark --model bge-m3 --eval evals/data/medquad-eval-v1.jsonl
python -m ragmed_retrieval benchmark --model bge-m3 --eval evals/data/medquad-eval-v1.jsonl --hybrid
python -m ragmed_retrieval benchmark --model bge-m3 --eval evals/data/medquad-eval-v1.jsonl \
    --device cpu --threads 4     # measure latency the way the server will run it

python -m ragmed_retrieval report                                      # results table
python -m ragmed_retrieval build-eval --out evals/data/medquad-eval-v1.jsonl   # rebuild the test set
```

Model presets: `bge-m3` (default), `qwen3-0.6b`, `minilm`, `medcpt`, `embeddinggemma`, `qwen3-4b`.

## Conventions

- Python 3.12, type hints everywhere, `mypy --strict` must pass.
- Tests live next to each package in `tests/` and never need network access or model downloads.
- Configuration comes only from environment variables; see `.env.example`.
