# ADR 0001: Embedding model and how it runs

- **Status:** accepted
- **Date:** 2026-11-20
- **Evidence:** [README › Embedding benchmark](../../README.md#embedding-benchmark),
  [`backend/evals/`](../../backend/evals/)

## Context

Retrieval quality caps answer quality: the language model can only cite passages that retrieval
finds. The embedding model decides which passages are found. It must also run on the production
server, an Oracle Cloud Always Free Arm VM with 4 CPU cores, 24 GB of RAM and no GPU, at a cost of
$0.

The original prototype used all-MiniLM-L6-v2. The plan expected Qwen3-Embedding-0.6B to replace
it, but required the choice to be made by a benchmark on our own data, not by leaderboard rank.

## Options considered

Four models were run on 1,000 held-out MedQuAD questions against 19,020 passages, each in full
precision on the same data and scoring code:

| Model | nDCG@10 | Recall@5 | CPU time per question (4 threads) |
| --- | ---: | ---: | ---: |
| BGE-M3 | 0.751 | 0.856 | 58 ms (PyTorch), 10 ms (ONNX int8) |
| Qwen3-Embedding-0.6B | 0.748 | 0.853 | 75 ms (PyTorch) |
| all-MiniLM-L6-v2 | 0.624 | 0.757 | 4 ms |
| MedCPT | 0.468 | 0.586 | not measured |

A paired bootstrap over the questions shows BGE-M3 and Qwen3 are tied (gap +0.3 points, 95%
interval −1.3 to +1.8). Both beat MiniLM by about 12.5 points, which is a real difference.

Hosted embedding APIs (Gemini, Voyage, Cloudflare Workers AI) were also considered. They add a
network round trip, a free-tier quota that can run out, a dependency on the provider keeping the
model available, and send patients' text to a third party. Local embedding is fast enough that
none of that is worth it.

## Decision

1. **Use BGE-M3** (BAAI, MIT licence). It ties Qwen3 on accuracy, is faster on CPU, and supports
   more than 100 languages.
2. **Embed passages once, offline, in full precision**, on any machine with a GPU, and store the
   vectors in the vector database.
3. **Embed questions on the server with an int8 ONNX export of BGE-M3**, run by ONNX Runtime.
   The export includes BGE-M3's pooling and normalisation, so the server needs only
   `onnxruntime` and `tokenizers`, not PyTorch.
4. **Dense search only for now.** Equal-weight hybrid search with BM25 lowered BGE-M3 by 7.3
   points.

## Consequences

- Question embedding drops from 58 ms to 10 ms on 4 threads, and the model from 2.3 GB to 569
  MB. Retrieval accuracy does not drop: int8 questions against full-precision passages scored
  nDCG@10 0.756 against 0.751.
- The server image needs no PyTorch, which keeps it about 1 GB smaller.
- Re-indexing the knowledge base needs a GPU machine (about 10 minutes on Apple Silicon) or a
  slow CPU run. That happens only when the corpus or the model changes.
- If the model is ever changed, every passage must be re-embedded: vectors from different models
  are not comparable.
- Not yet measured on the Oracle VM itself; the Apple M4 Pro numbers above are expected to be
  2–3× faster than the server's cores.

## How to reproduce

```bash
cd backend
make install-bench
make benchmark-all     # every experiment, about an hour on an Apple Silicon GPU
make export-onnx       # the int8 model the server uses
make significance      # confidence intervals
```
