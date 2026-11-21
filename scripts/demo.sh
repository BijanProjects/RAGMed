#!/usr/bin/env bash
# Run the whole app locally: three backend services and the web app.
# Usage: make demo   (from the repository root), then open http://localhost:3000
#        DEMO_CALENDAR=memory make demo   to book into a built-in calendar instead of Cal.com
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
for port in 3000 8000 8001 9001 9002; do
  if lsof -ti "tcp:$port" -sTCP:LISTEN > /dev/null 2>&1; then
    fail "port $port is in use (an earlier demo?). Stop it with: lsof -ti tcp:$port -sTCP:LISTEN | xargs kill"
  fi
done

# Real bookings go to Cal.com when a key is present; otherwise a built-in calendar is used.
CALENDAR="${DEMO_CALENDAR:-}"
if [[ -z "$CALENDAR" ]]; then
  if grep -q "^CAL_API_KEY=." "$ROOT/.env" 2>/dev/null; then CALENDAR=calcom; else CALENDAR=memory; fi
fi
# Fresh random secrets for each run: the clinician page reads appointments with one, the web
# app authenticates to the Coordinator with another, and browser sessions are signed with a third.
token() { "$PY" -c 'import secrets; print(secrets.token_urlsafe(32))'; }
CLINICIAN_TOKEN="$(token)"
WEB_TOKEN="$(token)"
SESSION_SECRET="$(token)"

# Postgres (saved conversations) and Valkey (rate limits) in Docker, when Docker is available.
DATABASE_URL="" VALKEY_URL=""
if command -v docker > /dev/null && docker info > /dev/null 2>&1; then
  docker compose -f "$ROOT/infra/compose/dev.yml" up -d --wait > /dev/null 2>&1 \
    && DATABASE_URL="postgresql://ragmed:ragmed-local-only@127.0.0.1:5432/ragmed" \
    && VALKEY_URL="redis://127.0.0.1:6379/0" \
    || echo "  note: could not start Postgres and Valkey; conversations and limits stay in memory"
else
  echo "  note: Docker is not running; conversations and rate limits stay in memory"
fi
ACCESS_CODE="${CLINICIAN_ACCESS_CODE:-clinic-demo}"

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
  (cd "$BACKEND" && RAGMED_LOG_LEVEL=WARNING RAGMED_PORT="$port" "$PY" -m "$module" "$@" \
    > "$LOGS/$name.log" 2>&1) &
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
RAGMED_CALENDAR="$CALENDAR" RAGMED_CLINICIAN_TOKENS="{\"web\": \"$CLINICIAN_TOKEN\"}" \
  start mcp-scheduling mcp_scheduling 9002
start diagnosis-agent diagnosis_agent 8001
RAGMED_DATABASE_URL="$DATABASE_URL" RAGMED_VALKEY_URL="$VALKEY_URL" \
  RAGMED_SERVICE_TOKENS="{\"web\": \"$WEB_TOKEN\"}" \
  start coordinator-agent coordinator_agent 8000

echo "  calendar: $CALENDAR · conversations: $([[ -n $DATABASE_URL ]] && echo postgres || echo memory) · limits: $([[ -n $VALKEY_URL ]] && echo valkey || echo memory)"
echo "  clinician view: http://localhost:3000/clinician (code: $ACCESS_CODE)"
if grep -q '^RAGMED_SMTP_PASSWORD=.' "$ROOT/.env" 2> /dev/null; then
  echo "  booking codes: emailed"
else
  echo "  booking codes: not emailed (no RAGMED_SMTP_* in .env); find them with"
  echo "      grep 'verification code' $LOGS/coordinator-agent.log"
fi
echo "  starting web app          → http://localhost:3000  (Ctrl-C stops everything)"
cd "$FRONTEND"
# Turnstile keys from .env when present (localhost must be one of the widget's hostnames),
# otherwise Cloudflare's published test keys, which always pass.
env_value() { sed -n "s/^$1=//p" "$ROOT/.env" 2> /dev/null | tail -1 | tr -d "\"'"; }
CHECK_SOURCE="keys set in the shell"
[[ -z ${NEXT_PUBLIC_TURNSTILE_SITE_KEY:-} ]] && CHECK_SOURCE="your Cloudflare Turnstile keys from .env"
NEXT_PUBLIC_TURNSTILE_SITE_KEY="${NEXT_PUBLIC_TURNSTILE_SITE_KEY:-$(env_value NEXT_PUBLIC_TURNSTILE_SITE_KEY)}"
TURNSTILE_SECRET_KEY="${TURNSTILE_SECRET_KEY:-$(env_value TURNSTILE_SECRET_KEY)}"
if [[ -n $NEXT_PUBLIC_TURNSTILE_SITE_KEY && -n $TURNSTILE_SECRET_KEY ]]; then
  echo "  bot check: $CHECK_SOURCE"
else
  echo "  bot check: Cloudflare's test keys (always pass)"
fi
COORDINATOR_URL=http://127.0.0.1:8000 COORDINATOR_TOKEN="$WEB_TOKEN" \
  SCHEDULING_URL=http://127.0.0.1:9002 \
  CLINICIAN_TOKEN="$CLINICIAN_TOKEN" CLINICIAN_ACCESS_CODE="$ACCESS_CODE" \
  NEXT_PUBLIC_TURNSTILE_SITE_KEY="${NEXT_PUBLIC_TURNSTILE_SITE_KEY:-1x00000000000000000000AA}" \
  TURNSTILE_SECRET_KEY="${TURNSTILE_SECRET_KEY:-1x0000000000000000000000000000000AA}" \
  SESSION_SECRET="$SESSION_SECRET" \
  NEXT_TELEMETRY_DISABLED=1 npm run dev -- --port 3000
