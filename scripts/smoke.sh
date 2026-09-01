#!/usr/bin/env bash
# curl one-liners get mangled in transit; run this instead.
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
: "${MODEL:?set MODEL in .env}"

echo "== vllm direct =="
curl -sS --max-time 120 "${VLLM_URL:-http://localhost:8000/v1}/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"paged attention is\",\"max_tokens\":32}" \
  || echo "vllm not answering on ${VLLM_URL:-http://localhost:8000/v1}"

echo
echo "== gateway =="
curl -sS --max-time 120 localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"paged attention is","max_tokens":64}' \
  || echo "gateway not answering on :8080 (expected until ./init.sh up)"
echo
