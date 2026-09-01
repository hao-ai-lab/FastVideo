#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS=8
export RVM_SP_SIZE=4
CONFIG="examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml"
OUTPUT="${RVM_8GPU_SMOKE_OUTPUT:-outputs/rvm_h3/8gpu_topology_smoke}"
RUN_NAME="${RVM_8GPU_SMOKE_RUN_NAME:-rvm-h3-faithful-8gpu-topology-smoke}"

# Phase 1: one update and a complete distributed checkpoint.
run_rvm_training \
    "${CONFIG}" \
    --method.samples_per_prompt 8 \
    --method.prompt_groups_per_rollout 4 \
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
    --method.prompt_groups_per_rollout 4 \
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
Eight-GPU topology gate completed:
  topology: SP4 x DP2
  K: 8
  global prompt groups: 4
  local rollout workload per DP replica: 2 prompts x 8 = 16 videos
  output: ${OUTPUT}
Inspect both W&B runs, checkpoint-2, the exported LoRA manifest, all reward
components, batch-global reward std, gradient clipping, and validation media.
EOF
