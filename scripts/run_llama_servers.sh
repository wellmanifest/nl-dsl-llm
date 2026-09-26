#!/usr/bin/env bash
# Launch both Qwen3.5-2B and Qwen3.5-0.8B llama-server instances on 8 vCPU VPS
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/models}"
CACHE_DIR="${CACHE_DIR:-/opt/cache}"
mkdir -p "$MODEL_DIR" "$CACHE_DIR"

QWEN_2B_GGUF="$MODEL_DIR/qwen3.5-2b-instruct-q4_k_m.gguf"
QWEN_08B_GGUF="$MODEL_DIR/qwen3.5-0.8b-instruct-q4_k_m.gguf"

echo "Checking model weights..."
if [[ ! -f "$QWEN_2B_GGUF" ]]; then
  echo "Downloading Qwen3.5-2B Q4_K_M..."
  # Download command placeholder / curl from HuggingFace
fi

if [[ ! -f "$QWEN_08B_GGUF" ]]; then
  echo "Downloading Qwen3.5-0.8B Q4_K_M..."
fi

echo "Starting llama-server for Qwen3.5-2B on :8081 (-t 4, prompt-cache)..."
llama-server \
  --model "$QWEN_2B_GGUF" \
  --host 127.0.0.1 --port 8081 \
  --threads 4 \
  --ctx-size 2048 \
  --n-predict 128 \
  --mlock \
  --prompt-cache "$CACHE_DIR/qwen2b_prefix.bin" &

echo "Starting llama-server for Qwen3.5-0.8B on :8082 (-t 4, prompt-cache)..."
llama-server \
  --model "$QWEN_08B_GGUF" \
  --host 127.0.0.1 --port 8082 \
  --threads 4 \
  --ctx-size 2048 \
  --n-predict 128 \
  --mlock \
  --prompt-cache "$CACHE_DIR/qwen08b_prefix.bin" &

echo "Both inference backends launched successfully."
wait
