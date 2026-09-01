#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS=8
export RVM_SP_SIZE=4
CONFIG="${RVM_SCALEUP_CONFIG:-examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml}"
STEPS="${RVM_SCALEUP_STEPS:-50}"
PROMPT_GROUPS="${RVM_SCALEUP_PROMPT_GROUPS:-8}"
EVAL_PROMPTS="${RVM_SCALEUP_EVAL_PROMPTS:-32}"
SELECTED_LR="${RVM_SELECTED_LR:-1e-5}"
MIN_TRAIN_PROMPTS="${RVM_SCALEUP_MIN_TRAIN_PROMPTS:-4096}"
OUTPUT="${RVM_SCALEUP_OUTPUT:-outputs/rvm_h3/8gpu_scaleup_pilot}"
RUN_NAME="${RVM_SCALEUP_RUN_NAME:-rvm-h3-faithful-8gpu-scaleup-pilot}"

python - "${RVM_TRAIN_DATA}" "${MIN_TRAIN_PROMPTS}" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
root = Path(sys.argv[1])
minimum = int(sys.argv[2])
files = sorted(root.rglob("*.parquet"))
rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
print(f"scale-up training prompts: {rows}")
if rows < minimum:
    raise RuntimeError(
        f"Scale-up requires at least {minimum} encoded prompts, found {rows}. "
        "Re-run 02_prepare_dataset.sh with a larger RVM_MAX_TRAIN_PROMPTS."
    )
PY

run_rvm_training \
    "${CONFIG}" \
    --method.samples_per_prompt 8 \
    --method.prompt_groups_per_rollout "${PROMPT_GROUPS}" \
    --method.optimizer_updates_per_rollout 2 \
    --method.positive_only_steps 0 \
    --method.validation.num_prompts "${EVAL_PROMPTS}" \
    --method.validation.every_steps 0 \
    --method.validation.log_sample_limit 16 \
    --training.optimizer.learning_rate "${SELECTED_LR}" \
    --training.loop.max_train_steps "${STEPS}" \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.checkpoint.training_state_checkpointing_steps 10 \
    --training.checkpoint.checkpoints_total_limit 6 \
    --training.tracker.run_name "${RUN_NAME}"

cat <<EOF
Scale-up pilot completed:
  config: ${CONFIG}
  LR: ${SELECTED_LR}
  optimizer steps: ${STEPS}
  prompt groups per collection: ${PROMPT_GROUPS}
  K: 8
  held-out prompts per 5%-interval evaluation: ${EVAL_PROMPTS}
  output: ${OUTPUT}
Select a checkpoint from held-out rewards and media; do not automatically use
the last checkpoint. Require non-decreasing VideoAlign TA/MQ, useful global
reward variance, low clipping frequency, and understood DT saturation.
EOF
