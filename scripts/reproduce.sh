#!/usr/bin/env bash
# Reproduce the committed speculative-decoding sweeps.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS="${CONFIGS:-configs/qwen3-8b-0.6b.json configs/qwen3-4b-0.6b.json}"
LLAMA_BIN="${SPECLAB_LLAMA_BIN:-$HOME/llama.cpp/src-b11050/build/bin}"

for CONFIG in $CONFIGS; do
  python3 -m specdec fetch --config "$CONFIG"
  python3 -m specdec run --config "$CONFIG" --llama-bin "$LLAMA_BIN" --out-dir results
done
