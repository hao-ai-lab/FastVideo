#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${FASTH3_MODEL_DIR}"
require_path "${RVM_SMOKE_DATA}"
require_path "${RVM_EVAL_DATA}"
require_path "${VIDEOALIGN_RUNTIME_PATH}"
require_path "${VIDEOALIGN_CHECKPOINT_PATH}"

python -m py_compile \
    fastvideo/train/methods/rl/rvm.py \
    fastvideo/train/methods/rl/rvm_faithful.py \
    fastvideo/train/methods/rl/rvm_local_metrics.py \
    fastvideo/train/methods/rl/common/rvm_utils.py \
    fastvideo/train/methods/rl/common/minimax_h3_rvm.py \
    fastvideo/train/methods/rl/rewards/media.py \
    fastvideo/train/methods/rl/rewards/dynamic_tracking.py \
    fastvideo/train/models/minimax_h3/minimax_h3_rvm.py \
    fastvideo/train/entrypoint/export_rvm_lora.py \
    examples/train/rvm_h3/prepare_prompts.py \
    examples/train/rvm_h3/preflight_rewards.py \
    examples/train/rvm_h3/infer_lora.py

pytest -q \
    fastvideo/tests/train/methods/test_rvm_utils.py \
    fastvideo/tests/train/methods/test_rvm_reward_diagnostics.py \
    fastvideo/tests/train/methods/test_minimax_h3_dmd2.py \
    fastvideo/tests/train/methods/test_rvm_configs.py \
    fastvideo/tests/inference/lora/test_merge_lora_math.py

# This loads all five production rewards and verifies that the RAFT term ranks
# a moving synthetic video above an otherwise matching static one.
python examples/train/rvm_h3/preflight_rewards.py --device cuda

# Build the actual H3/FSDP/LoRA/reward config without taking an optimizer step.
NUM_GPUS=1 bash examples/train/run.sh \
    examples/train/configs/rl/minimax_h3/rvm_h3_1gpu_smoke.yaml \
    --dry-run

echo "One-GPU preflight passed. Run 04_run_1gpu_smoke.sh next."
