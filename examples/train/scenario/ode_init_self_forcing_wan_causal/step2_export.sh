#!/usr/bin/env bash
# Step 2 — Export ODE-init weights from KD checkpoint to diffusers format.
#
# Usage:
#   bash examples/train/scenario/ode_init_self_forcing_wan_causal/step2_export.sh

set -euo pipefail

CHECKPOINT_DIR="${1:-outputs/kdtest/kd_causal/checkpoint-300}"
OUTPUT_DIR="${2:-outputs/kdtest/kd_ode_init}"

echo "Exporting ODE-init weights..."
echo "  checkpoint: ${CHECKPOINT_DIR}"
echo "  output:     ${OUTPUT_DIR}"

python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role student \
    --checkpoint "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}"

echo "Done. Use the exported weights in step3_self_forcing.yaml."
