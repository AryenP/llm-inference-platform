#!/usr/bin/env bash
# Long pastes arrive mangled by bracketed paste; run this instead of a one-liner.
# Usage: ./scripts/pull_models.sh [dest]        (default ~/models)
#        WITH_BF16=1 ./scripts/pull_models.sh   also pulls the 16.4 GB BF16 weights
set -euo pipefail

dest="${1:-$HOME/models}"
mkdir -p "$dest"

hf="hf"
command -v hf >/dev/null 2>&1 || hf="uvx --from huggingface_hub hf"

$hf download Qwen/Qwen3-8B-AWQ       --local-dir "$dest/qwen3-8b-awq"
$hf download BAAI/bge-m3             --local-dir "$dest/bge-m3"
$hf download BAAI/bge-reranker-v2-m3 --local-dir "$dest/bge-reranker"

# BF16 is 16.4 GB of weights — only where there is VRAM to serve it
if [ "${WITH_BF16:-0}" = "1" ]; then
  $hf download Qwen/Qwen3-8B --local-dir "$dest/qwen3-8b"
fi

echo
echo "models in $dest — set MODEL in .env to $dest/qwen3-8b-awq"
