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
    echo "Custom-node topology smoke supports NUM_GPUS=8 or 16." >&2
    exit 2
fi
if [[ "${RVM_SP_SIZE}" != "4" ]] || (( NUM_GPUS % RVM_SP_SIZE != 0 )); then
    echo "Use RVM_SP_SIZE=4 for the 8/16-H100 custom-node topology gate." >&2
    exit 2
fi
DP_REPLICAS=$((NUM_GPUS / RVM_SP_SIZE))
GLOBAL_PROMPT_GROUPS=$((2 * DP_REPLICAS))

CONFIG="examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml"
OUTPUT="${RVM_TOPOLOGY_SMOKE_OUTPUT:-outputs/rvm_h3/${NUM_GPUS}gpu_topology_smoke}"
RUN_NAME="${RVM_TOPOLOGY_SMOKE_RUN_NAME:-rvm-h3-faithful-${NUM_GPUS}gpu-topology-smoke}"

# Phase 1: one update and a complete distributed checkpoint.
run_rvm_training \
    "${CONFIG}" \
    --method.samples_per_prompt 8 \
    --method.prompt_groups_per_rollout "${GLOBAL_PROMPT_GROUPS}" \
    --method.optimizer_updates_per_rollout 1 \
    --method.validation.num_prompts 8 \
    --method.validation.log_sample_limit 8 \
    --method.validation.every_steps 1 \
    --training.loop.max_train_steps 1 \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.checkpoint.training_state_checkpointing_steps 1 \
    --training.checkpoint.checkpoints_total_limit 3 \
    --training.tracker.run_name "${RUN_NAME}-phase1"

# Phase 2: reload optimizer/scheduler/RNG state and advance one more update.
run_rvm_training \
    "${CONFIG}" \
    --method.samples_per_prompt 8 \
    --method.prompt_groups_per_rollout "${GLOBAL_PROMPT_GROUPS}" \
    --method.optimizer_updates_per_rollout 1 \
    --method.validation.num_prompts 8 \
    --method.validation.log_sample_limit 8 \
    --method.validation.every_steps 1 \
    --training.loop.max_train_steps 2 \
    --training.checkpoint.output_dir "${OUTPUT}" \
    --training.checkpoint.training_state_checkpointing_steps 1 \
    --training.checkpoint.checkpoints_total_limit 3 \
    --training.checkpoint.resume_from_checkpoint latest \
    --training.tracker.run_name "${RUN_NAME}-resume"

bash examples/train/rvm_h3/09_export_lora.sh \
    "${CONFIG}" \
    "${OUTPUT}/checkpoint-2" \
    "${OUTPUT}/fasth3_rvm_topology_smoke.safetensors"

cat <<EOF
Custom-node topology gate completed:
  GPUs: ${NUM_GPUS} H100s
  topology: SP4 x DP${DP_REPLICAS}
  K: 8
  global prompt groups: ${GLOBAL_PROMPT_GROUPS}
  local rollout workload per DP replica: 2 prompts x 8 = 16 videos
  output: ${OUTPUT}
Inspect both W&B runs, checkpoint-2, the exported LoRA manifest, all reward
components, batch-global reward std, gradient clipping, and validation media.
EOF
