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
SELECTED_LR="${RVM_SELECTED_LR:-1e-5}"

for variant in exact audio_anchor full_anchor; do
    config="examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_${variant}.yaml"
    run_rvm_training \
        "${config}" \
        --training.optimizer.learning_rate "${SELECTED_LR}" \
        --training.loop.max_train_steps "${ANCHOR_STEPS}" \
        --training.checkpoint.output_dir "outputs/rvm_h3/anchor_${variant}" \
        --training.tracker.run_name "rvm-h3-anchor-${variant}"
done

cat <<'EOF'
Decision rule:
  - exact wins only if its video gain is larger without audio degradation;
  - audio_anchor is the default safety/performance candidate;
  - full_anchor wins if unanchored video changes visibly damage broad quality.
Use the same fixed validation prompts and seeds for all three runs.
EOF
