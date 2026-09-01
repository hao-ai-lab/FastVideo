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

# FastVideo selects its supported torch build through UV_TORCH_BACKEND.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
bash examples/train/rvm_h3/00_install_current_env.sh

echo "Environment ready: conda activate ${RVM_ENV_NAME}"
