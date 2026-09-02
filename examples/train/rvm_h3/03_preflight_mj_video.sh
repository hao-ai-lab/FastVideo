#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

require_path "${MJ_VIDEO_RUNTIME_PATH}/scripts/model/moe_reward.py"
require_path "${MJ_VIDEO_MODEL_PATH}/model.safetensors"
require_path "${MJ_VIDEO_MODEL_PATH}/.fastvideo_revision"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}/config.json"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}/model.safetensors"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}/.fastvideo_revision"

python -m py_compile \
    fastvideo/train/methods/rl/rewards/calibration.py \
    fastvideo/train/methods/rl/rewards/mj_video_compat.py \
    fastvideo/train/methods/rl/rewards/mj_video.py \
    fastvideo/train/methods/rl/rvm_reward_profile.py \
    examples/train/rvm_h3/preflight_mj_video.py \
    examples/train/rvm_h3/calibrate_reward_profile.py

pytest -q \
    fastvideo/tests/train/methods/test_reward_calibration.py \
    fastvideo/tests/train/methods/test_reward_calibration_cli.py \
    fastvideo/tests/train/methods/test_mj_video_reward.py \
    fastvideo/tests/train/methods/test_rvm_reward_diagnostics.py \
    fastvideo/tests/train/methods/test_rvm_configs.py

python examples/train/rvm_h3/preflight_mj_video.py \
    --device "${MJ_VIDEO_PREFLIGHT_DEVICE:-cuda}" \
    --output "${MJ_VIDEO_PREFLIGHT_OUTPUT:-${RVM_REWARD_ROOT}/mj_video_preflight.json}"

echo "MJ-VIDEO real-checkpoint preflight passed."
