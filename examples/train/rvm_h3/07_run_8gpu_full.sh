#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
CONFIG="${RVM_FULL_CONFIG:-examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_full.yaml}"
SELECTED_LR="${RVM_SELECTED_LR:-1e-5}"
VIDEO_ANCHOR="${RVM_VIDEO_ANCHOR_BETA:-0.0}"
AUDIO_ANCHOR="${RVM_AUDIO_ANCHOR_BETA:-1e-3}"

run_rvm_training \
    "${CONFIG}" \
    --training.optimizer.learning_rate "${SELECTED_LR}" \
    --method.video_anchor_beta "${VIDEO_ANCHOR}" \
    --method.audio_anchor_beta "${AUDIO_ANCHOR}" \
    --training.checkpoint.output_dir "${RVM_FULL_OUTPUT:-outputs/rvm_h3/8gpu_full}" \
    --training.tracker.run_name "${RVM_FULL_RUN_NAME:-rvm-h3-8gpu-full}"

# For 180 optimizer steps, validation/checkpointing occurs every 9 steps (5%).
python examples/train/rvm_h3/11_collect_results.py \
    --root outputs/rvm_h3 \
    --output outputs/rvm_h3/results_index.json
