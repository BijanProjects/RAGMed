#!/usr/bin/env bash
# Deploy the backend to the Oracle server:  make deploy   (after make infra)
#
# 1. Creates .env.production once (git-ignored): random tokens for every link between services,
#    plus the API keys copied from .env. The same file holds what Vercel needs.
# 2. Copies the committed code (git archive, so local experiments never ship) and the search
#    index and ONNX model (rsync, only what changed) to the server.
# 3. Builds and starts everything with Docker Compose and waits until it is healthy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env.production"
KEY="$HOME/.ssh/ragmed_oci"
API_HOST="${API_HOST:-ragmed-api.bijanroudini.com}"
WEB_HOST="${WEB_HOST:-ragmed.bijanroudini.com}"

IP="$(terraform -chdir="$ROOT/infra/terraform" output -raw public_ip 2> /dev/null)" \
  || { echo "✗ no server yet: run make infra first"; exit 1; }
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "ubuntu@$IP")
[[ -z $(git -C "$ROOT" status --porcelain) ]] \
  || echo "  note: uncommitted changes are not deployed (only the last commit is)"

secret() { python3 -c 'import secrets; print(secrets.token_urlsafe(32))'; }
from_env() { sed -n "s/^$1=//p" "$ROOT/.env" | tail -1 | tr -d "\"'"; }

# Generated once and kept: changing them would log everyone out and break the Vercel settings.
if [[ ! -f $ENV_FILE ]]; then
  umask 077
  {
    echo "# Production secrets for RAGMed. Never commit. Created by scripts/deploy.sh."
    echo "API_HOST=$API_HOST"
    echo "WEB_HOST=$WEB_HOST"
    echo "DATA_DIR=/opt/ragmed/data"
    echo "POSTGRES_PASSWORD=$(secret)"
    for name in TOKEN_MEDICAL TOKEN_DIAGNOSIS TOKEN_SCHEDULING TOKEN_CLINICIAN TOKEN_WEB SESSION_SECRET; do
      echo "$name=$(secret)"
    done
    echo "CLINICIAN_ACCESS_CODE=$(python3 -c 'import secrets; print(secrets.token_hex(6))')"
  } > "$ENV_FILE"
  echo "  created .env.production with new secrets"
fi
# API keys always follow .env, so a rotated key only needs a redeploy.
for name in GROQ_API_KEY CAL_API_KEY RAGMED_SMTP_USERNAME RAGMED_SMTP_PASSWORD \
  NEXT_PUBLIC_TURNSTILE_SITE_KEY TURNSTILE_SECRET_KEY; do
  value="$(from_env "$name")"
  [[ -n $value ]] || { echo "✗ $name is missing from .env"; exit 1; }
  grep -v "^$name=" "$ENV_FILE" > "$ENV_FILE.tmp"; echo "$name=$value" >> "$ENV_FILE.tmp"
  mv "$ENV_FILE.tmp" "$ENV_FILE"
done
chmod 600 "$ENV_FILE"

echo "  waiting for the server's first-boot setup…"
"${SSH[@]}" "cloud-init status --wait > /dev/null; test -f /var/lib/cloud/ragmed-ready"

echo "  copying the code ($(git -C "$ROOT" rev-parse --short HEAD))"
"${SSH[@]}" "rm -rf /opt/ragmed/app && mkdir -p /opt/ragmed/app /opt/ragmed/data/artifacts/clinic"
git -C "$ROOT" archive --format=tar HEAD | "${SSH[@]}" "tar -x -C /opt/ragmed/app"

echo "  copying the knowledge base, search index and model (first time ~800 MB)"
RSYNC=(rsync -az --delete --exclude .lock -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new")
DATA="$ROOT/backend/data"
"${RSYNC[@]}" "$DATA/raw/" "ubuntu@$IP:/opt/ragmed/data/raw/"
"${RSYNC[@]}" "$DATA/artifacts/onnx/" "ubuntu@$IP:/opt/ragmed/data/artifacts/onnx/"
"${RSYNC[@]}" "$DATA/artifacts/qdrant/" "ubuntu@$IP:/opt/ragmed/data/artifacts/qdrant/"
scp -q -i "$KEY" "$ENV_FILE" "ubuntu@$IP:/opt/ragmed/.env.production"
# The containers run as the owner of the data folder (see APP_UID in prod.yml).
"${SSH[@]}" "chmod 600 /opt/ragmed/.env.production && \
  printf 'APP_UID=%s\nAPP_GID=%s\n' \$(id -u) \$(id -g) >> /opt/ragmed/.env.production"

echo "  building and starting (first time ~5 minutes)"
"${SSH[@]}" "cd /opt/ragmed/app && docker compose -f infra/compose/prod.yml \
  --env-file /opt/ragmed/.env.production up -d --build --remove-orphans --wait --wait-timeout 600" \
  2>&1 | grep -vE "^\s*$|Pulling|Waiting|Downloading|Extracting|Verifying|Download complete|Pull complete" || true
"${SSH[@]}" "cd /opt/ragmed/app && docker compose -f infra/compose/prod.yml \
  --env-file /opt/ragmed/.env.production ps --format '  {{.Service}}: {{.Status}}'"
echo "  server: $IP   API: https://$API_HOST (needs the DNS record)"
