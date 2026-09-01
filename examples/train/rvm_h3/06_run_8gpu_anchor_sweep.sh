#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
ANCHOR_STEPS="${RVM_ANCHOR_SWEEP_STEPS:-20}"
ANCHOR_GROUPS="${RVM_ANCHOR_SWEEP_PROMPT_GROUPS:-8}"
ANCHOR_EVAL_PROMPTS="${RVM_ANCHOR_SWEEP_EVAL_PROMPTS:-32}"
ANCHOR_LOG_VIDEOS="${RVM_ANCHOR_SWEEP_LOG_VIDEOS:-8}"
SELECTED_LR="${RVM_SELECTED_LR:-1e-5}"

if (( ANCHOR_STEPS < 2 )); then
    echo "RVM_ANCHOR_SWEEP_STEPS must be at least 2." >&2
    exit 1
fi
if (( ANCHOR_GROUPS % (NUM_GPUS / RVM_SP_SIZE) != 0 )); then
    echo "Prompt groups must divide evenly across DP replicas." >&2
    exit 1
fi

for variant in exact audio_anchor full_anchor; do
    config="examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_${variant}.yaml"
    run_rvm_training \
        "${config}" \
        --training.optimizer.learning_rate "${SELECTED_LR}" \
        --training.loop.max_train_steps "${ANCHOR_STEPS}" \
        --method.prompt_groups_per_rollout "${ANCHOR_GROUPS}" \
        --method.validation.num_prompts "${ANCHOR_EVAL_PROMPTS}" \
        --method.validation.log_sample_limit "${ANCHOR_LOG_VIDEOS}" \
        --method.validation.every_steps 0 \
        --training.checkpoint.output_dir "outputs/rvm_h3/anchor_${variant}" \
        --training.checkpoint.training_state_checkpointing_steps "${ANCHOR_STEPS}" \
        --training.tracker.run_name "rvm-h3-faithful-anchor-${variant}"
done

cat <<'EOF'
Decision rule:
  - exact is the published RVM reference and wins only if audio remains intact;
  - audio_anchor is the H3 safety candidate if it preserves audio without
    materially reducing video reward gains;
  - full_anchor wins only if broad video quality drifts under the other two.
Use the same fixed validation prompts, seeds, LR, and rollout budget for all
three runs. Validation follows the automatic 5%-of-progress cadence.
EOF
