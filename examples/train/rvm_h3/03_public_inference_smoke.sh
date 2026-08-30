#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${FASTH3_MODEL_DIR}"

OUTPUT_DIR="${RVM_ARTIFACT_ROOT}/inference_smoke"
mkdir -p "${OUTPUT_DIR}"
PROMPT="${RVM_SMOKE_PROMPT:-integrated_multimodal_description: [Shot 1] Live-action, cinematic. A red race car drives along a wet mountain road while the camera tracks beside it at steady speed. overall_soundscape: Tire noise on wet asphalt, wind, and a distant engine echo. non_diegetic_music: N/A}"

python examples/inference/basic/fasth3.py \
    --model-path "${FASTH3_MODEL_DIR}" \
    --prompt "${PROMPT}" \
    --output "${OUTPUT_DIR}" \
    --profile strict \
    --height 480 \
    --width 832 \
    --num-frames 124 \
    --steps 5 \
    --seed 1000 \
    --repeats 1 \
    --num-gpus 1 \
    --vsa-sparsity 0.9 \
    --vsa-tile-size 64 \
    --vsa-kernel triton \
    --no-fa4 \
    --no-h3-fusions \
    --no-inference-torch-compile \
    --no-compile-vae \
    --no-parallel-vae

echo "Strict FastH3 inference output: ${OUTPUT_DIR}"
