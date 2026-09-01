#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
SWEEP_STEPS="${RVM_LR_SWEEP_STEPS:-8}"
SWEEP_GROUPS="${RVM_LR_SWEEP_PROMPT_GROUPS:-4}"
SWEEP_EVAL_PROMPTS="${RVM_LR_SWEEP_EVAL_PROMPTS:-32}"
SWEEP_LOG_VIDEOS="${RVM_LR_SWEEP_LOG_VIDEOS:-8}"
IFS=',' read -r -a LEARNING_RATES <<< "${RVM_LR_SWEEP:-5e-6,1e-5,2e-5}"

if (( SWEEP_STEPS < 2 )); then
    echo "RVM_LR_SWEEP_STEPS must be at least 2." >&2
    exit 1
fi
if (( SWEEP_GROUPS % (NUM_GPUS / RVM_SP_SIZE) != 0 )); then
    echo "Prompt groups must divide evenly across DP replicas." >&2
    exit 1
fi

for learning_rate in "${LEARNING_RATES[@]}"; do
    tag="$(echo "${learning_rate}" | tr '.-' '__')"
    run_rvm_training \
        examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml \
        --training.optimizer.learning_rate "${learning_rate}" \
        --training.loop.max_train_steps "${SWEEP_STEPS}" \
        --method.prompt_groups_per_rollout "${SWEEP_GROUPS}" \
        --method.validation.num_prompts "${SWEEP_EVAL_PROMPTS}" \
        --method.validation.log_sample_limit "${SWEEP_LOG_VIDEOS}" \
        --method.validation.every_steps 0 \
        --training.checkpoint.output_dir "outputs/rvm_h3/lr_${tag}" \
        --training.checkpoint.training_state_checkpointing_steps "${SWEEP_STEPS}" \
        --training.tracker.run_name "rvm-h3-faithful-lr-${tag}"
done

cat <<'EOF'
The sweep uses paper-faithful, unanchored RVM: batch-global reward standard
deviation and continuous Uniform(0,1) training times. Validation follows the
required automatic 5%-of-optimizer-progress cadence on a bounded 32-prompt set;
the full run uses 100 prompts. Select the largest LR with improving held-out
rewards, low clipping frequency, non-saturated useful reward variance, and no
qualitative collapse. The Wan paper's 5e-5 LR may be added through RVM_LR_SWEEP
only after the lower bracket is stable on 35B FastH3.
EOF
