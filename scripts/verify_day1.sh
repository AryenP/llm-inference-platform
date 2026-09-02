#!/usr/bin/env bash
# Day 1 acceptance check. Prints one report block; paste the whole thing back.
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
: "${MODEL:?set MODEL in .env}"

VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
GW="${GW:-http://localhost:8080}"

echo "===== versions ====="
echo "commit   : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "vllm     : $(uv run vllm --version 2>/dev/null || echo 'not found')"
echo "postgres : $(psql --version 2>/dev/null || echo 'not found')"
echo "sampler  : flashinfer=${VLLM_USE_FLASHINFER_SAMPLER:-unset} util=${GPU_MEM_UTIL:-unset}"

echo
echo "===== postgres ====="
psql "${DATABASE_URL:-postgresql://rag:rag@localhost:5432/rag}" -tAc \
  "select extname || ' ' || extversion from pg_extension order by extname" \
  2>/dev/null || echo "cannot connect to ${DATABASE_URL:-postgresql://rag:rag@localhost:5432/rag}"

echo
echo "===== vllm ====="
curl -sf --max-time 10 "$VLLM_URL/models" 2>/dev/null |
  python3 -c "import sys,json;print('serving:', ', '.join(m['id'] for m in json.load(sys.stdin)['data']))" \
  || echo "not answering at $VLLM_URL"

echo
echo "===== gateway ====="
curl -sf --max-time 10 "$GW/health" 2>/dev/null || echo "no /health at $GW"

echo
echo "===== /query ====="
resp=$(curl -sf --max-time 180 "$GW/query" -H 'Content-Type: application/json' \
  -d '{"prompt":"paged attention is","max_tokens":64}' 2>/dev/null) || resp=""

if [ -z "$resp" ]; then
  echo "no response from $GW/query"
  exit 1
fi

printf '%s' "$resp" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('text      :', ' '.join(d['text'].split())[:140])
print('model     :', d['model'])
print('n_chunks  :', d['n_chunks'])
print('ttft_ms   :', d['ttft_ms'])
print('total_ms  :', d['total_ms'])
r = d['ttft_ms'] / d['total_ms'] if d['total_ms'] else 1.0
# a streamed 64-token completion should spend most of its time decoding, so ttft
# ought to be a small slice of total; a ratio near 1 means something buffered it
print('ttft/total:', round(r, 3), 'PASS' if r < 0.5 else 'SUSPECT - response looks buffered')
"
