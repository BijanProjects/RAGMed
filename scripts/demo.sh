#!/usr/bin/env bash
# Run the whole app locally: three backend services and the web app.
# Usage: make demo   (from the repository root), then open http://localhost:3000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
LOGS="$BACKEND/data/artifacts/logs"
PY="$BACKEND/.venv/bin/python"

fail() { echo "✗ $*" >&2; exit 1; }

# Prerequisites, each with the command that fixes it.
[[ -x "$PY" ]] || fail "backend not installed: cd backend && make install && make install-bench"
[[ -f "$BACKEND/data/artifacts/onnx/BAAI__bge-m3/model_int8.onnx" ]] \
  || fail "no int8 model: cd backend && make export-onnx"
[[ -f "$BACKEND/data/artifacts/qdrant/medquad-index.json" ]] \
  || fail "knowledge base not loaded: cd backend && make mcp-ingest"
grep -q "^GROQ_API_KEY=." "$ROOT/.env" 2>/dev/null || [[ -n "${GROQ_API_KEY:-}" ]] \
  || fail "GROQ_API_KEY missing: add it to $ROOT/.env"
[[ -d "$FRONTEND/node_modules" ]] || (cd "$FRONTEND" && npm install --no-fund --no-audit)

mkdir -p "$LOGS"
PIDS=()
cleanup() {
  echo; echo "Stopping…"
  kill "${PIDS[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start() {  # name, module, port, extra args
  local name=$1 module=$2 port=$3; shift 3
  (cd "$BACKEND" && RAGMED_LOG_LEVEL=WARNING "$PY" -m "$module" "$@" > "$LOGS/$name.log" 2>&1) &
  PIDS+=($!)
  printf "  starting %-18s" "$name"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/healthz" > /dev/null; then echo "✓ http://127.0.0.1:$port"; return; fi
    sleep 0.5
  done
  echo "✗ (see $LOGS/$name.log)"; exit 1
}

echo "RAGMed demo"
start mcp-medical mcp_medical 9001 serve
start diagnosis-agent diagnosis_agent 8001
start coordinator-agent coordinator_agent 8000

echo "  starting web app          → http://localhost:3000  (Ctrl-C stops everything)"
cd "$FRONTEND"
COORDINATOR_URL=http://127.0.0.1:8000 NEXT_TELEMETRY_DISABLED=1 npm run dev -- --port 3000
