#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

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
    vllm serve "$MODEL" --port 8000 --max-model-len 8192 &
    uv run uvicorn app.main:app --port 8080 --reload
    ;;
  test)   uv run pytest -q ;;
  eval)   uv run python -m eval.run --golden eval/golden.jsonl ;;
  bench)  uv run python -m bench.sweep --out results.json ;;
  down)   pkill -f "vllm.entrypoints" || true; service postgresql stop 2>/dev/null || true ;;
  *) echo "usage: ./init.sh [up|test|eval|bench|down]"; exit 1 ;;
esac
