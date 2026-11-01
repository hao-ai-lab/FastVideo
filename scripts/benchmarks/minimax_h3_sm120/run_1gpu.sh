#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
mode="${1:-}"
case "${mode}" in
    native) enabled=0 ;;
    sm120) enabled=1 ;;
    *) echo "Usage: bash scripts/benchmarks/minimax_h3_sm120/run_1gpu.sh native|sm120 [example arguments]" >&2; exit 2 ;;
esac
shift
model_path="${MODEL_PATH:-/path/to/FastH3-8-Step-V2}"
if [[ ! -d "${model_path}" ]]; then
    echo "Model directory does not exist: ${model_path}" >&2
    exit 1
fi
output_root="${repo_dir}/../outputs"
profile_dir="${output_root}/profiles/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${profile_dir}"
cd -- "${repo_dir}"
export PYTHONPATH="${repo_dir}/fastvideo-kernel/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FASTVIDEO_H3_VSA_SM120="${enabled}"
export FASTVIDEO_H3_VSA_SM89_KERNEL=original
export FASTVIDEO_H3_VSA_FINE_ATTENTION=""
export FASTVIDEO_H3_VSA_FP4=0
export FASTVIDEO_H3_VSA_TILE_FIRST=0
export FASTVIDEO_H3_VSA_HEADS_FIRST_TILE=0
export FASTVIDEO_MINIMAX_H3_EXACT_KERNELS=0
export FASTVIDEO_H3_ENCODER_SPLIT=0
export FASTVIDEO_H3_ENCODER_LAYERWISE=1
export FASTVIDEO_H3_ADALN_CACHE=1
export FASTVIDEO_H3_ADALN_TABLE=""
export FASTVIDEO_H3_ADALN_DUMP=""
export FASTVIDEO_H3_PARK_MODULES=vae,audio_vae

{
    echo "SM120 comparison mode: ${mode}; FASTVIDEO_H3_VSA_SM120=${enabled}"
    echo "Model: ${model_path}"
    echo "Placement: layerwise text encoder; DiT stays on GPU between requests; VAEs offloaded to CPU"
    echo "Log: ${profile_dir}/${mode}.log"
    git rev-parse HEAD
    git diff --stat
    python - <<'PY'
import torch
import triton
import fastvideo_kernel.block_sparse_attn_sm120 as kernel

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    raise RuntimeError("The paired model benchmark requires an SM120 CUDA GPU")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; Triton: {triton.__version__}")
print(f"SM120 kernel source: {kernel.__file__}")
PY
    python -u examples/inference/basic/basic_fasth3_8step.py \
        --model-path "${model_path}" \
        --prompt "A red fox walks slowly through a snowy forest, leaving footprints in fresh snow. A stationary wide shot shows the fox's full body and detailed fur. Soft wind and crisp footsteps are audible." \
        --output "${output_root}/fasth3_sm120/${mode}" \
        --num-gpus 1 --execution-backend mp --profile strict \
        --vsa-kernel triton --no-fa4 --replicated-dit \
        --no-torch-compile --no-inference-torch-compile --no-compile-vae \
        --offload-text-encoder --offload-vae --h3-sequential-load --no-lazy-module-load \
        --no-parallel-vae --height 768 --width 1344 --num-frames 360 \
        --warmup --warmup-seed 999 --seed 42 --repeats 3 "$@"
} 2>&1 | tee "${profile_dir}/${mode}.log"
