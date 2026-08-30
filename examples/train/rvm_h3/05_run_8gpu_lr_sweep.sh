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
IFS=',' read -r -a LEARNING_RATES <<< "${RVM_LR_SWEEP:-5e-6,1e-5,2e-5}"

for learning_rate in "${LEARNING_RATES[@]}"; do
    tag="$(echo "${learning_rate}" | tr '.-' '__')"
    run_rvm_training \
        examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_audio_anchor.yaml \
        --training.optimizer.learning_rate "${learning_rate}" \
        --training.loop.max_train_steps "${SWEEP_STEPS}" \
        --training.checkpoint.output_dir "outputs/rvm_h3/lr_${tag}" \
        --training.checkpoint.training_state_checkpointing_steps "${SWEEP_STEPS}" \
        --training.tracker.run_name "rvm-h3-lr-${tag}"
done

cat <<'EOF'
Select the largest LR that has finite gradients, no visible collapse, stable
independent validation rewards, and no audio regression. Do not select solely
by rollout reward. The default full-run candidate is 1e-5.
EOF
