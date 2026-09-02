#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
if [[ "${NUM_GPUS}" != "8" && "${NUM_GPUS}" != "16" ]]; then
    echo "The custom-node full campaign supports NUM_GPUS=8 or 16." >&2
    exit 2
fi
if [[ "${RVM_SP_SIZE}" != "4" ]] || (( NUM_GPUS % RVM_SP_SIZE != 0 )); then
    echo "Use RVM_SP_SIZE=4 for the 8/16-H100 full campaign." >&2
    exit 2
fi
if [[ "${RVM_FULL_APPROVED:-0}" != "1" ]]; then
    echo "Set RVM_FULL_APPROVED=1 only after topology, LR, reward-profile, anchor, and scale-up gates pass." >&2
    exit 1
fi

CONFIG="${RVM_FULL_CONFIG:-examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_full.yaml}"
SELECTED_LR="${RVM_SELECTED_LR:-1e-5}"
VIDEO_ANCHOR="${RVM_VIDEO_ANCHOR_BETA:-0.0}"
AUDIO_ANCHOR="${RVM_AUDIO_ANCHOR_BETA:-0.0}"
MIN_TRAIN_PROMPTS="${RVM_FULL_MIN_TRAIN_PROMPTS:-48000}"
FULL_STEPS="${RVM_FULL_STEPS:-180}"
FULL_PROMPT_GROUPS="${RVM_FULL_PROMPT_GROUPS:-32}"
FULL_K="${RVM_FULL_K:-8}"
FULL_EVAL_PROMPTS="${RVM_FULL_EVAL_PROMPTS:-100}"
CHECKPOINT_INTERVAL="$(( (FULL_STEPS + 19) / 20 ))"
DP_REPLICAS=$((NUM_GPUS / RVM_SP_SIZE))
OUTPUT="${RVM_FULL_OUTPUT:-outputs/rvm_h3/${NUM_GPUS}gpu_full}"
RUN_NAME="${RVM_FULL_RUN_NAME:-rvm-h3-faithful-${NUM_GPUS}gpu-full}"

if (( FULL_PROMPT_GROUPS % DP_REPLICAS != 0 )); then
    echo "RVM_FULL_PROMPT_GROUPS=${FULL_PROMPT_GROUPS} must be divisible by DP replicas=${DP_REPLICAS}." >&2
    exit 2
fi
if (( FULL_K < 2 || FULL_EVAL_PROMPTS < 1 || FULL_EVAL_PROMPTS > 100 )); then
    echo "Full campaign requires K>=2 and 1<=eval prompts<=100." >&2
    exit 2
fi
if grep -q 'mjvideo_' "${CONFIG}"; then
    require_path "${MJ_VIDEO_RUNTIME_PATH}"
    require_path "${MJ_VIDEO_MODEL_PATH}"
    require_path "${MJ_VIDEO_BASE_MODEL_PATH}"
    require_path "${MJ_VIDEO_CALIBRATION_PATH}"
fi

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
        "Encode the complete pinned RVM/VidProM training split before launch."
    )
PY

run_rvm_training \
    "${CONFIG}" \
    --method.samples_per_prompt "${FULL_K}" \
    --method.prompt_groups_per_rollout "${FULL_PROMPT_GROUPS}" \
    --method.optimizer_updates_per_rollout 2 \
    --method.positive_only_steps 0 \
    --method.validation.num_prompts "${FULL_EVAL_PROMPTS}" \
    --method.validation.every_steps 0 \
    --training.optimizer.learning_rate "${SELECTED_LR}" \
    --method.video_anchor_beta "${VIDEO_ANCHOR}" \
    --method.audio_anchor_beta "${AUDIO_ANCHOR}" \
    --training.loop.max_train_steps "${FULL_STEPS}" \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.checkpoint.training_state_checkpointing_steps "${CHECKPOINT_INTERVAL}" \
    --training.tracker.run_name "${RUN_NAME}"

python examples/train/rvm_h3/11_collect_results.py \
    --root outputs/rvm_h3 \
    --output outputs/rvm_h3/results_index.json

cat <<EOF
Full custom-node campaign completed:
  GPUs/topology: ${NUM_GPUS} H100s, SP4 x DP${DP_REPLICAS}
  config: ${CONFIG}
  steps: ${FULL_STEPS}
  prompt groups/K: ${FULL_PROMPT_GROUPS} x ${FULL_K}
  validation prompts: ${FULL_EVAL_PROMPTS}
  checkpoint/eval interval: ${CHECKPOINT_INTERVAL}
  output: ${OUTPUT}
EOF
