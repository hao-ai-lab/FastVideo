#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not installed or not on PATH." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${RVM_ENV_NAME}"; then
    conda create -y -n "${RVM_ENV_NAME}" python=3.11
fi
conda activate "${RVM_ENV_NAME}"
python -m pip install --upgrade pip uv

# FastVideo pins the supported torch build. Select CUDA 13 on Blackwell by
# default; set UV_TORCH_BACKEND=cu126 for supported CUDA-12 clusters.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
uv pip install -e ".[eval,test]"

# Reward-only dependencies. Do not install GenRL's requirements.txt: it pins
# an old torch/transformers stack and would silently replace FastVideo's tested
# runtime. We reuse only the public reward implementations and checkpoints.
uv pip install \
    "hpsv3==1.0.0" \
    "qwen-vl-utils" \
    "trl>=0.18" \
    "liger-kernel" \
    "decord" \
    "safetensors"

python - <<'PY'
import torch
import diffusers
import transformers
import ptlflow
import hpsv3
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "diffusers": diffusers.__version__,
    "transformers": transformers.__version__,
    "ptlflow": getattr(ptlflow, "__version__", "unknown"),
    "hpsv3": getattr(hpsv3, "__version__", "installed"),
})
PY

echo "Environment ready: conda activate ${RVM_ENV_NAME}"
