# RAGMed backend

All Python code: data ingestion, retrieval, evaluation, and (soon) the MCP servers and agents.

## Layout

```text
backend/
├─ packages/                  reusable libraries, imported by the services
│  ├─ ragmed_core/            settings, logging, trace ids — shared by every service
│  └─ ragmed_retrieval/       corpus loading, chunking, embeddings, retrieval metrics
├─ services/                  deployable programs: MCP servers and agents (coming next)
├─ data/raw/                  the MedQuAD corpus (input, never modified)
├─ data/artifacts/            generated files such as cached embeddings (git-ignored)
├─ evals/                     the retrieval benchmark
│  ├─ data/                   the frozen set of 1,000 test questions
│  ├─ results/                one JSON file per model and pipeline
│  ├─ analysis/               confidence intervals and per-question-type scores
│  └─ analysis.py             recomputes analysis/ from the cached vectors
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

### `ragmed_retrieval`

Read in this order; each module builds on the previous one.

| Module | Purpose |
| --- | --- |
| `corpus.py` | Loads MedQuAD, cleans text, removes duplicates, gives each answer a stable id |
| `chunking.py` | Splits long answers into passages of at most 384 words, at sentence boundaries |
| `embedding.py` | Turns text into vectors; one interface, any sentence-transformers model behind it, plus MedCPT's separate query and article encoders |
| `metrics.py` | Scores a ranking: Recall@k, nDCG@k, MRR |
| `evalset.py` | Builds the fixed set of 1,000 test questions with their correct answers |
| `hybrid.py` | BM25 keyword search, Reciprocal Rank Fusion, and rerankers (Qwen3-Reranker, bge-reranker) |
| `benchmark.py` | Embeds the corpus, searches for every test question (dense, hybrid, or reranked), and scores the results |
| `cli.py` | Command-line entry point for the steps above |
| `testing.py` | A tiny fake embedder so tests run without downloading models |

## Commands

```bash
make help          # list everything
make install       # set up .venv and git hooks
make install-bench # add PyTorch and the model libraries (needed for search and benchmarks)
make check         # what CI runs: ruff, mypy --strict, pytest with coverage
make lock          # re-pin dependencies after editing requirements/*.in

make search Q="What are the symptoms of asthma?"   # ask the knowledge base (BGE-M3, CPU)
make report        # print the benchmark results table
make significance  # recompute the confidence intervals
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
