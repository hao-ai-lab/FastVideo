#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

hf_download() {
    if command -v hf >/dev/null 2>&1; then
        hf download "$@"
    else
        huggingface-cli download "$@"
    fi
}

mkdir -p "${RVM_REWARD_ROOT}"

# These values must match fastvideo/train/methods/rl/rewards/mj_video.py.
# They are intentionally not environment-overridable for reported runs.
MJ_VIDEO_SOURCE_COMMIT="cc1d2c9587a620e9ebd3599ae4cdd21b5fd7c87a"
MJ_VIDEO_MODEL_REVISION="5d32c2416bf5ffb9331a175890744e73defb54c4"
MJ_VIDEO_BASE_REVISION="e4f6747"

if [[ ! -d "${MJ_VIDEO_RUNTIME_PATH}/.git" ]]; then
    rm -rf "${MJ_VIDEO_RUNTIME_PATH}"
    git clone https://github.com/aiming-lab/MJ-Video.git "${MJ_VIDEO_RUNTIME_PATH}"
fi
git -C "${MJ_VIDEO_RUNTIME_PATH}" fetch --all --tags
git -C "${MJ_VIDEO_RUNTIME_PATH}" checkout --detach "${MJ_VIDEO_SOURCE_COMMIT}"

mkdir -p "${MJ_VIDEO_MODEL_PATH}" "${MJ_VIDEO_BASE_MODEL_PATH}"
hf_download \
    MJ-Bench/MJ-VIDEO-2B \
    --revision "${MJ_VIDEO_MODEL_REVISION}" \
    --local-dir "${MJ_VIDEO_MODEL_PATH}"
printf '%s\n' "${MJ_VIDEO_MODEL_REVISION}" \
    >"${MJ_VIDEO_MODEL_PATH}/.fastvideo_revision"

hf_download \
    OpenGVLab/InternVL2-2B \
    --revision "${MJ_VIDEO_BASE_REVISION}" \
    --local-dir "${MJ_VIDEO_BASE_MODEL_PATH}"
printf '%s\n' "${MJ_VIDEO_BASE_REVISION}" \
    >"${MJ_VIDEO_BASE_MODEL_PATH}/.fastvideo_revision"

require_path "${MJ_VIDEO_RUNTIME_PATH}/scripts/model/moe_reward.py"
require_path "${MJ_VIDEO_MODEL_PATH}/model.safetensors"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}/config.json"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}/model.safetensors"

OBSERVED_SOURCE="$(git -C "${MJ_VIDEO_RUNTIME_PATH}" rev-parse HEAD)"
if [[ "${OBSERVED_SOURCE}" != "${MJ_VIDEO_SOURCE_COMMIT}" ]]; then
    echo "MJ-VIDEO source revision mismatch: expected ${MJ_VIDEO_SOURCE_COMMIT}, got ${OBSERVED_SOURCE}." >&2
    exit 1
fi
if [[ "$(<"${MJ_VIDEO_MODEL_PATH}/.fastvideo_revision")" != "${MJ_VIDEO_MODEL_REVISION}" ]]; then
    echo "MJ-VIDEO checkpoint revision marker mismatch." >&2
    exit 1
fi
if [[ "$(<"${MJ_VIDEO_BASE_MODEL_PATH}/.fastvideo_revision")" != "${MJ_VIDEO_BASE_REVISION}" ]]; then
    echo "InternVL2 base revision marker mismatch." >&2
    exit 1
fi

cat <<EOF
Downloaded:
  MJ-VIDEO code:       ${MJ_VIDEO_RUNTIME_PATH} @ ${MJ_VIDEO_SOURCE_COMMIT}
  MJ-VIDEO checkpoint: ${MJ_VIDEO_MODEL_PATH} @ ${MJ_VIDEO_MODEL_REVISION}
  InternVL2 base:      ${MJ_VIDEO_BASE_MODEL_PATH} @ ${MJ_VIDEO_BASE_REVISION}
EOF
