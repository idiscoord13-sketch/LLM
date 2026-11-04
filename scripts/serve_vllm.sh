#!/usr/bin/env bash
# Serve a merged model with vLLM (OpenAI-compatible). Linux/WSL2 + NVIDIA GPU.
# Install vLLM in its OWN environment (it pins torch):  python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm
# Keep vLLM on localhost and expose only the gateway (which adds auth, rate limits, consent logging).
set -euo pipefail
MODEL="${1:-out/sft-merged}"
PORT="${PORT:-8000}"
: "${UPSTREAM_API_KEY:?set UPSTREAM_API_KEY (shared secret between gateway and vLLM)}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name candidate \
  --host 127.0.0.1 --port "$PORT" \
  --api-key "$UPSTREAM_API_KEY" \
  --dtype bfloat16 \
  --max-model-len "${MAX_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching
