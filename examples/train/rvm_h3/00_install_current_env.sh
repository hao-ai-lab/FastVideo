#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Install the exact FastH3 RVM runtime into the active Python environment.
# This script is intentionally container/cloud agnostic. The conda bootstrap
# calls it after creating the environment; Modal and other container runtimes
# call it with RVM_SKIP_CONDA=1.
activate_rvm_env

python -m pip install --upgrade pip uv

uv_args=()
if [[ "${RVM_SKIP_CONDA:-0}" == "1" ]]; then
    uv_args+=(--system)
fi

uv pip install "${uv_args[@]}" -e ".[eval,test]"
uv pip install "${uv_args[@]}" \
    "decord" \
    "accelerate>=1.1" \
    "fire" \
    "liger-kernel" \
    "qwen-vl-utils" \
    "safetensors" \
    "trl==0.8.6"
uv pip install "${uv_args[@]}" --no-deps "hpsv3==1.0.0"

python - <<'PY'
import diffusers
import hpsv3
import ptlflow
import torch
import transformers

print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "ptlflow": getattr(ptlflow, "__version__", "unknown"),
        "hpsv3": getattr(hpsv3, "__version__", "installed"),
    }
)
PY
