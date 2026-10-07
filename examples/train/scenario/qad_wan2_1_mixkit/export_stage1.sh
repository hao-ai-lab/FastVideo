#!/usr/bin/env bash
# Export a modular-trainer DCP checkpoint for stage-2 initialization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT_DIR=${1:-checkpoints/wan_t2v_qat_finetune/checkpoint-4000}
OUTPUT_DIR=${2:-checkpoints/wan_t2v_qat_finetune/diffusers}

python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role student \
    --checkpoint "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}"
