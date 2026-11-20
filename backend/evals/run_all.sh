#!/usr/bin/env bash
# Run every retrieval experiment in the README, fastest first, then recompute the analysis.
# Run from backend/:  make benchmark-all   (progress: tail -f data/artifacts/benchmark.log)
#
# The first run of each embedding model embeds all 19,020 passages (10-20 minutes on an Apple
# Silicon GPU) and caches the vectors; later runs only embed the 1,000 test questions.
set -euo pipefail

EVAL=evals/data/medquad-eval-v1.jsonl
LOG=data/artifacts/benchmark.log
PY=.venv/bin/python
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

run() {
  echo "=== $*  (started $(date +%H:%M:%S))" >> "$LOG"
  "$PY" -m ragmed_retrieval benchmark --eval "$EVAL" "$@" 2>&1 \
    | grep --line-buffered -E "reranked|nDCG|Error|Traceback" >> "$LOG"
}

# Embedding models, with and without BM25 keyword search
for model in minilm qwen3-0.6b bge-m3 medcpt; do
  run --model "$model"
  run --model "$model" --hybrid
done

# The int8 ONNX export the server will run (make export-onnx first)
if [[ -f data/artifacts/onnx/BAAI__bge-m3/model_int8.onnx ]]; then
  run --model bge-m3 --runtime onnx
fi

# Rerankers on top of BGE-M3, smallest first
run --model bge-m3 --rerank minilm-ce
run --model bge-m3 --rerank minilm-ce --rerank-depth 20
run --model bge-m3 --rerank bge-reranker-base
run --model bge-m3 --rerank bge-reranker-v2-m3

echo "=== analysis  (started $(date +%H:%M:%S))" >> "$LOG"
"$PY" evals/analysis.py >> "$LOG" 2>&1
echo "=== all done $(date +%H:%M:%S)" >> "$LOG"
