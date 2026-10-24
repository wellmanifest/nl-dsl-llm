#!/usr/bin/env bash
# Launch single-resident llama-server instance on 8 vCPU / 8 GB RAM VPS.
# Enforces the invariant: only ONE model resident in RAM at a time, preserving >= 5.6 GB RAM for host services.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/models}"
CACHE_DIR="${CACHE_DIR:-/opt/cache}"
PROFILE="${1:-qwen3-1.7b}"
PORT="${PORT:-8081}"
THREADS="${THREADS:-4}"

# Emit LiteLLM configuration if requested
if [[ "${1:-}" == "--litellm-config" ]]; then
  cat <<'EOF'
model_list:
  - model_name: nl-router
    litellm_params:
      model: openai/qwen3-1.7b-instruct
      api_base: http://127.0.0.1:8081/v1
      api_key: none
      temperature: 0.2
      max_tokens: 128
      timeout: 10
  - model_name: nl-quality
    litellm_params:
      model: openai/qwen3.5-2b-instruct
      api_base: http://127.0.0.1:8081/v1
      api_key: none
      temperature: 0.2
      max_tokens: 160
      timeout: 15
  - model_name: nl-ollama
    litellm_params:
      model: ollama/qwen2.5:1.5b
      api_base: http://127.0.0.1:11434
      timeout: 10
router_settings:
  timeout: 15
general_settings:
  master_key: sk-paxlet-local-dev
EOF
  exit 0
fi

mkdir -p "$MODEL_DIR" "$CACHE_DIR"

case "$PROFILE" in
  qwen3-1.7b|qwen3_1.7b)
    MODEL_FILE="$MODEL_DIR/qwen3-1.7b-instruct-q4_k_m.gguf"
    MODEL_NAME="Qwen3-1.7B-Instruct (Primary Default Router)"
    CTX=4096
    PREDICT=128
    ;;
  qwen3.5-2b|qwen3.5_2b)
    MODEL_FILE="$MODEL_DIR/qwen3.5-2b-instruct-q4_k_m.gguf"
    MODEL_NAME="Qwen3.5-2B (Quality Multilingual Fallback)"
    CTX=4096
    PREDICT=160
    ;;
  granite-4.0-micro|granite_4.0_micro)
    MODEL_FILE="$MODEL_DIR/granite-4.0-micro-3b-instruct-q4_k_m.gguf"
    MODEL_NAME="Granite-4.0-Micro-3B (Enterprise Tools & JSON)"
    CTX=4096
    PREDICT=128
    ;;
  smollm3-3b|smollm3_3b)
    MODEL_FILE="$MODEL_DIR/smollm3-3b-instruct-q4_k_m.gguf"
    MODEL_NAME="SmolLM3-3B (Permissive Tool Calling)"
    CTX=4096
    PREDICT=128
    ;;
  lfm2.5-1.2b|lfm2.5_1.2b)
    MODEL_FILE="$MODEL_DIR/lfm2.5-1.2b-instruct-q4_k_m.gguf"
    MODEL_NAME="LFM2.5-1.2B-Instruct (Ultra-Fast CPU Edge)"
    CTX=2048
    PREDICT=96
    ;;
  qwen3.5-0.8b|qwen3.5_0.8b)
    MODEL_FILE="$MODEL_DIR/qwen3.5-0.8b-instruct-q4_k_m.gguf"
    MODEL_NAME="Qwen3.5-0.8B (Ultra-Fast Suggestion Router)"
    CTX=2048
    PREDICT=64
    ;;
  *)
    echo "Usage: $0 [qwen3-1.7b|qwen3.5-2b|granite-4.0-micro|smollm3-3b|lfm2.5-1.2b|qwen3.5-0.8b|--litellm-config]"
    exit 1
    ;;
esac

echo "======================================================================"
echo "Starting Single-Resident Model Server"
echo "Profile:  $MODEL_NAME"
echo "Target:   $MODEL_FILE"
echo "Port:     $PORT | Threads: $THREADS | CTX: $CTX"
echo "Memory:   Enforcing single-resident invariant (leaves >= 5.6 GB RAM free)"
echo "======================================================================"

# Terminate any previously running llama-server to ensure only 1 model is resident in RAM
if pgrep -f "llama-server.*--port $PORT" >/dev/null 2>&1; then
  echo "Terminating previous resident llama-server on port $PORT..."
  pkill -f "llama-server.*--port $PORT" || true
  sleep 1
fi

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Warning: Model weights not found at $MODEL_FILE."
  echo "To download, run:"
  echo "  huggingface-cli download <repo> <filename> --local-dir $MODEL_DIR"
  echo "Or verify local Ollama / LiteLLM gateway on http://127.0.0.1:11434"
  exit 0
fi

exec llama-server \
  --model "$MODEL_FILE" \
  --host 127.0.0.1 --port "$PORT" \
  --threads "$THREADS" \
  --ctx-size "$CTX" \
  --n-predict "$PREDICT" \
  --mlock \
  --prompt-cache "$CACHE_DIR/${PROFILE}_prefix.bin"
