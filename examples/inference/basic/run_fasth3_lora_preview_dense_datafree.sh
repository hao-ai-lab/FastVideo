#!/usr/bin/env bash
set -euo pipefail

repo="FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA"
adapter="dense-datafree/adapter_model.safetensors"
adapter_path="$(hf download "$repo" "$adapter")"

python examples/inference/basic/basic_fasth3_lora_preview.py \
  --lora-path "$adapter_path" \
  --lora-strength "${FASTH3_LORA_STRENGTH:-1.0}" \
  --output "${FASTH3_LORA_OUTPUT:-outputs/fasth3_lora_preview/dense-datafree}" \
  "$@" \
  --no-vsa
