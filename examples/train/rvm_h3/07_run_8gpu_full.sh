#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
if [[ "${NUM_GPUS}" != 8 || "${RVM_SP_SIZE}" != 4 ]]; then
    echo "The validated full topology is NUM_GPUS=8, RVM_SP_SIZE=4 (SP4 x DP2)." >&2
    exit 1
fi
if [[ "${RVM_FULL_APPROVED:-0}" != 1 ]]; then
    echo "Set RVM_FULL_APPROVED=1 only after the topology, LR, anchor, and scale-up gates pass." >&2
    exit 1
fi

CONFIG="${RVM_FULL_CONFIG:-examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_full.yaml}"
SELECTED_LR="${RVM_SELECTED_LR:-1e-5}"
VIDEO_ANCHOR="${RVM_VIDEO_ANCHOR_BETA:-0.0}"
AUDIO_ANCHOR="${RVM_AUDIO_ANCHOR_BETA:-0.0}"
MIN_TRAIN_PROMPTS="${RVM_FULL_MIN_TRAIN_PROMPTS:-10000}"

python - "${RVM_TRAIN_DATA}" "${MIN_TRAIN_PROMPTS}" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
root = Path(sys.argv[1])
minimum = int(sys.argv[2])
files = sorted(root.rglob("*.parquet"))
rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
print(f"full-run training prompts: {rows}")
if rows < minimum:
    raise RuntimeError(
        f"Full RVM run requires at least {minimum} encoded prompts, found {rows}. "
        "Use the complete 48,998-prompt RVM/VidProM bank when storage permits."
    )
PY

run_rvm_training \
    "${CONFIG}" \
    --training.optimizer.learning_rate "${SELECTED_LR}" \
    --method.video_anchor_beta "${VIDEO_ANCHOR}" \
    --method.audio_anchor_beta "${AUDIO_ANCHOR}" \
    --training.checkpoint.output_dir "${RVM_FULL_OUTPUT:-outputs/rvm_h3/8gpu_full}" \
    --training.tracker.run_name "${RVM_FULL_RUN_NAME:-rvm-h3-faithful-8gpu-full}"

# For 180 optimizer steps, validation/checkpointing occurs every 9 steps (5%).
python examples/train/rvm_h3/11_collect_results.py \
    --root outputs/rvm_h3 \
    --output outputs/rvm_h3/results_index.json
