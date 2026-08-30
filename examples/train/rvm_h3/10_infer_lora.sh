#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${FASTH3_MODEL_DIR}"

LORA_PATH="${1:?Usage: $0 <adapter.safetensors> [prompt] [output-dir]}"
PROMPT="${2:-integrated_multimodal_description: [Shot 1] Live-action, cinematic. A golden retriever runs through shallow ocean water at sunset while the camera tracks beside it. overall_soundscape: Splashes, light surf, and wind. non_diegetic_music: N/A}"
OUTPUT_DIR="${3:-outputs/rvm_h3/lora_inference}"
NUM_GPUS="${NUM_GPUS:-1}"
require_path "${LORA_PATH}"

python examples/train/rvm_h3/infer_lora.py \
    --model-path "${FASTH3_MODEL_DIR}" \
    --lora-path "${LORA_PATH}" \
    --lora-strength "${RVM_LORA_STRENGTH:-1.0}" \
    --prompt "${PROMPT}" \
    --output "${OUTPUT_DIR}" \
    --profile strict \
    --height 480 \
    --width 832 \
    --num-frames 124 \
    --steps 5 \
    --seed 1000 \
    --repeats 1 \
    --num-gpus "${NUM_GPUS}" \
    --vsa-sparsity 0.9 \
    --vsa-tile-size 64 \
    --vsa-kernel "${RVM_VSA_KERNEL:-triton}" \
    --no-fa4 \
    --no-h3-fusions \
    --no-inference-torch-compile \
    --no-compile-vae \
    --no-parallel-vae
