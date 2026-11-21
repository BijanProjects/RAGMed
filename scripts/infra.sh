#!/usr/bin/env bash
# Create or update the Oracle Cloud server with Terraform:  make infra
#   (plan only:  make infra ARGS=plan)
# Reads your tenancy from ~/.oci/config and allows SSH only from your current IP address.
# Free Arm servers are sometimes "out of capacity"; this retries every 2 minutes until one is free.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF="$ROOT/infra/terraform"
KEY="$HOME/.ssh/ragmed_oci"
ACTION="${1:-apply}"

[[ -f $HOME/.oci/config ]] || { echo "✗ ~/.oci/config is missing (see the README's deployment section)"; exit 1; }
[[ -f $KEY.pub ]] || ssh-keygen -q -t ed25519 -f "$KEY" -N "" -C ragmed-deploy

export TF_VAR_tenancy_ocid="$(sed -n 's/^tenancy=//p' "$HOME/.oci/config" | head -1)"
export TF_VAR_ssh_public_key="$(cat "$KEY.pub")"
export TF_VAR_ssh_cidr="$(curl -fsS https://api.ipify.org)/32"
echo "  SSH allowed from $TF_VAR_ssh_cidr"

terraform -chdir="$TF" init -input=false > /dev/null
if [[ $ACTION == plan ]]; then
  exec terraform -chdir="$TF" plan -input=false
fi

for attempt in $(seq 1 60); do
  if terraform -chdir="$TF" apply -input=false -auto-approve 2> "$TF/.last-error"; then
    terraform -chdir="$TF" output
    exit 0
  fi
  if grep -qi "out of host capacity\|OutOfCapacity" "$TF/.last-error"; then
    echo "  Montreal has no free Arm capacity right now (attempt $attempt); retrying in 2 minutes…"
    sleep 120
  else
    cat "$TF/.last-error"; exit 1
  fi
done
echo "✗ still no capacity after 2 hours; try again later"; exit 1
