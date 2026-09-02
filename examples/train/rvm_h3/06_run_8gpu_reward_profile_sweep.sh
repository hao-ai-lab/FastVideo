#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_EVAL_DATA}"
require_path "${MJ_VIDEO_RUNTIME_PATH}"
require_path "${MJ_VIDEO_MODEL_PATH}"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}"
require_path "${MJ_VIDEO_CALIBRATION_PATH}"

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
SELECTED_LR="${RVM_SELECTED_LR:-1e-5}"
SWEEP_STEPS="${RVM_REWARD_PROFILE_SWEEP_STEPS:-20}"
PROMPT_GROUPS="${RVM_REWARD_PROFILE_PROMPT_GROUPS:-8}"
SAMPLES_PER_PROMPT="${RVM_REWARD_PROFILE_K:-8}"
EVAL_PROMPTS="${RVM_REWARD_PROFILE_EVAL_PROMPTS:-32}"
EVAL_INTERVAL="$(( (SWEEP_STEPS + 1) / 2 ))"
CHECKPOINT_INTERVAL="${EVAL_INTERVAL}"

for value in \
    "${SWEEP_STEPS}" \
    "${PROMPT_GROUPS}" \
    "${SAMPLES_PER_PROMPT}" \
    "${EVAL_PROMPTS}"; do
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "Reward-profile sweep counts must be positive integers; got ${value}." >&2
        exit 2
    fi
done
if (( SAMPLES_PER_PROMPT < 2 )); then
    echo "RVM_REWARD_PROFILE_K must be at least two." >&2
    exit 2
fi
if (( EVAL_PROMPTS > 100 )); then
    echo "RVM_REWARD_PROFILE_EVAL_PROMPTS must be <= 100." >&2
    exit 2
fi
if (( PROMPT_GROUPS % (NUM_GPUS / RVM_SP_SIZE) != 0 )); then
    echo "Prompt groups must be divisible by the DP replica count." >&2
    exit 2
fi

if [[ "${RVM_SKIP_MJ_PREFLIGHT:-0}" != "1" ]]; then
    bash examples/train/rvm_h3/03_preflight_mj_video.sh
fi

profiles=(published_rvm physion_mj)
configs=(
    examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_exact.yaml
    examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_physion_mj.yaml
)

for index in "${!profiles[@]}"; do
    profile="${profiles[${index}]}"
    config="${configs[${index}]}"
    output="outputs/rvm_h3/reward_profile_${profile}"
    run_rvm_training \
        "${config}" \
        --method.samples_per_prompt "${SAMPLES_PER_PROMPT}" \
        --method.prompt_groups_per_rollout "${PROMPT_GROUPS}" \
        --method.validation.num_prompts "${EVAL_PROMPTS}" \
        --method.validation.every_steps "${EVAL_INTERVAL}" \
        --training.optimizer.learning_rate "${SELECTED_LR}" \
        --training.loop.max_train_steps "${SWEEP_STEPS}" \
        --training.checkpoint.output_dir "${output}" \
        --training.checkpoint.training_state_checkpointing_steps "${CHECKPOINT_INTERVAL}" \
        --training.tracker.run_name "rvm-h3-reward-profile-${profile}"
done

cat <<'EOF'
Reward-profile sweep complete.

The two runs share the same FastH3 initialization, RVM velocity objective,
continuous training-time distribution, LoRA, topology, prompt order, seeds,
learning rate, K, optimizer budget, and fixed validation prompts. They differ
only in the configured reward profile and the implementation class required to
broadcast calibrated diagnostics.

Select using paired held-out prompt deltas, raw and calibrated component values,
full-video inspection, audio preservation, and an independent evaluator. Do not
select from the final on-policy training batch or each profile's own aggregate
alone.
EOF
