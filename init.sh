#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

# vllm lands on PATH when installed with --system (the pod), or inside .venv when
# installed with uv sync --extra serve (WSL, where PEP 668 blocks --system)
vllm_cmd() {
  command -v vllm >/dev/null 2>&1 && echo "vllm" || echo "uv run vllm"
}

# Host workarounds, detected rather than configured. These belong here and NOT in
# .env: .env is recreated on the rented card, where both of these are wrong and
# would silently degrade the pod. Gate on the actual cause, not on "am I local".
gpu_host_env() {
  if grep -qi microsoft /proc/version 2>/dev/null; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    echo "wsl2 detected — VLLM_WSL2_ENABLE_PIN_MEMORY=1 (UVA pin-memory crash)"
  fi
  # FlashInfer JIT-compiles its sampling kernels and needs the full CUDA toolkit.
  # WSL2 ships the driver only, so there is no nvcc and no /usr/local/cuda.
  if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "no nvcc — VLLM_USE_FLASHINFER_SAMPLER=0 (native torch sampler)"
  fi
}

vllm_up() {
  curl -sf --max-time 2 "${VLLM_URL:-http://localhost:8000/v1}/models" >/dev/null 2>&1
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
    if curl -sf localhost:8000/health >/dev/null 2>&1; then
      echo "something is already serving on :8000 — run ./init.sh down first" >&2
      exit 1
    fi
    start_postgres
    gpu_host_env
    $(vllm_cmd) serve "$MODEL" --port 8000 --max-model-len "${MAX_MODEL_LEN:-8192}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" &
    vllm_pid=$!
    trap 'kill "$vllm_pid" 2>/dev/null || true' EXIT INT TERM
    # Cold start is ~65s here (weights, torch.compile, CUDA graph capture) while
    # uvicorn is up in under a second — without this wait, /query 502s for a minute.
    echo "waiting for vllm on :8000 …"
    until curl -sf localhost:8000/health >/dev/null 2>&1; do
      kill -0 "$vllm_pid" 2>/dev/null || { echo "vllm exited before becoming ready" >&2; exit 1; }
      sleep 2
    done
    echo "vllm ready — starting gateway on :8080"
    uv run uvicorn app.main:app --port 8080 --reload
    ;;
  serve)
    : "${MODEL:?set MODEL in .env}"
    gpu_host_env
    exec $(vllm_cmd) serve "$MODEL" --port 8000 --max-model-len "${MAX_MODEL_LEN:-8192}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}"
    ;;
  ingest) shift; uv run python -m app.ingest "$@" ;;
  gen)    shift; uv run python -m eval.generate "$@" ;;
  review) shift; uv run python -m eval.review "$@" ;;
  test)   uv run pytest -q ;;
  eval)   uv run python -m eval.run --golden eval/golden.jsonl ;;
  bench)  uv run python -m bench.sweep --out results.json ;;
  down)
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "vllm serve"       2>/dev/null || true
    service postgresql stop 2>/dev/null || true
    ;;
  *) echo "usage: ./init.sh [up|serve|ingest|gen|review|test|eval|bench|down]"; exit 1 ;;
esac
