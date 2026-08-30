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

mkdir -p "$(dirname "${FASTH3_MODEL_DIR}")" "${RVM_REWARD_ROOT}"

hf_download \
    FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
    --local-dir "${FASTH3_MODEL_DIR}"

VIDEOALIGN_COMMIT="aba26b658fec7d9fd30c295187b548ea673c8769"
if [[ ! -d "${VIDEOALIGN_RUNTIME_PATH}/.git" ]]; then
    rm -rf "${VIDEOALIGN_RUNTIME_PATH}"
    git clone https://github.com/ModelTC/VideoAlign.git "${VIDEOALIGN_RUNTIME_PATH}"
fi
git -C "${VIDEOALIGN_RUNTIME_PATH}" fetch --all --tags
git -C "${VIDEOALIGN_RUNTIME_PATH}" checkout --detach "${VIDEOALIGN_COMMIT}"

rm -rf "${VIDEOALIGN_CHECKPOINT_PATH}"
hf_download KwaiVGI/VideoReward --local-dir "${VIDEOALIGN_CHECKPOINT_PATH}"
# HPSv3RewardInferencer resolves from the HF cache; this command prefetches it.
hf_download MizzenAI/HPSv3 >/dev/null

require_path "${FASTH3_MODEL_DIR}/transformer"
require_path "${FASTH3_MODEL_DIR}/vae"
require_path "${FASTH3_MODEL_DIR}/text_encoder"
require_path "${VIDEOALIGN_RUNTIME_PATH}/inference.py"
require_path "${VIDEOALIGN_CHECKPOINT_PATH}"

cat <<EOF
Downloaded:
  FastH3:          ${FASTH3_MODEL_DIR}
  VideoAlign code: ${VIDEOALIGN_RUNTIME_PATH} @ ${VIDEOALIGN_COMMIT}
  VideoReward:     ${VIDEOALIGN_CHECKPOINT_PATH}
  HPSv3 cache:     ${HF_HOME}
EOF
