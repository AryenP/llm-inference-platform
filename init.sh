#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

case "${1:-up}" in
  up)
    : "${MODEL:?set MODEL in .env}"
    docker compose up -d postgres
    vllm serve "$MODEL" --port 8000 --max-model-len 8192 &
    uv run uvicorn app.main:app --port 8080 --reload
    ;;
  test)   uv run pytest -q ;;
  eval)   uv run python -m eval.run --golden eval/golden.jsonl ;;
  bench)  uv run python -m bench.sweep --out results.json ;;
  down)   docker compose down; pkill -f "vllm.entrypoints" || true ;;
  *) echo "usage: ./init.sh [up|test|eval|bench|down]"; exit 1 ;;
esac
