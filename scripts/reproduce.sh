#!/usr/bin/env bash
# Reproduce the committed speculative-decoding sweep.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/qwen3-4b-0.6b.json}"
LLAMA_BIN="${SPECLAB_LLAMA_BIN:-$HOME/llama.cpp/src-b11050/build/bin}"

python3 -m specdec fetch --config "$CONFIG"
python3 -m specdec run --config "$CONFIG" --llama-bin "$LLAMA_BIN" --out-dir results
