#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

# vllm lands on PATH when installed with --system (the pod), or inside .venv when
# installed with uv sync --extra serve (WSL, where PEP 668 blocks --system)
vllm_cmd() {
  command -v vllm >/dev/null 2>&1 && echo "vllm" || echo "uv run vllm"
}

start_postgres() {
  pg_isready -q 2>/dev/null && return 0
  service postgresql start 2>/dev/null || {
    echo "postgres not installed — run ./scripts/setup_postgres.sh" >&2
    exit 1
  }
  until pg_isready -q; do sleep 1; done
}

case "${1:-up}" in
  up)
    : "${MODEL:?set MODEL in .env}"
    start_postgres
    $(vllm_cmd) serve "$MODEL" --port 8000 --max-model-len 8192 \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" &
    uv run uvicorn app.main:app --port 8080 --reload
    ;;
  test)   uv run pytest -q ;;
  eval)   uv run python -m eval.run --golden eval/golden.jsonl ;;
  bench)  uv run python -m bench.sweep --out results.json ;;
  down)   pkill -f "vllm.entrypoints" || true; service postgresql stop 2>/dev/null || true ;;
  *) echo "usage: ./init.sh [up|test|eval|bench|down]"; exit 1 ;;
esac
