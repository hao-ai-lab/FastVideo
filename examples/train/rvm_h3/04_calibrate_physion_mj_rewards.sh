#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

require_path "${FASTH3_MODEL_DIR}"
require_path "${RVM_TRAIN_DATA}"
require_path "${RVM_PROMPT_DIR}/train_h3.txt"
require_path "${VIDEOALIGN_RUNTIME_PATH}"
require_path "${VIDEOALIGN_CHECKPOINT_PATH}"
require_path "${MJ_VIDEO_RUNTIME_PATH}"
require_path "${MJ_VIDEO_MODEL_PATH}"
require_path "${MJ_VIDEO_BASE_MODEL_PATH}"

export NUM_GPUS="${NUM_GPUS:-4}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-${NUM_GPUS}}"
CALIBRATION_VIDEOS="${RVM_CALIBRATION_VIDEOS:-100}"
CALIBRATION_ROOT="${RVM_CALIBRATION_ROOT:-outputs/rvm_h3/calibration_bank}"
CALIBRATION_VIDEO_DIR="${CALIBRATION_ROOT}/validation/step-000000"
CALIBRATION_CONFIG="examples/train/configs/rl/minimax_h3/h3_rvm_calibration_bank.yaml"

if ! [[ "${CALIBRATION_VIDEOS}" =~ ^[0-9]+$ ]] ||
   (( CALIBRATION_VIDEOS < 2 || CALIBRATION_VIDEOS > 100 )); then
    echo "RVM_CALIBRATION_VIDEOS must be an integer in [2, 100]." >&2
    exit 2
fi

video_count() {
    if [[ ! -d "${CALIBRATION_VIDEO_DIR}" ]]; then
        echo 0
        return
    fi
    find "${CALIBRATION_VIDEO_DIR}" \
        -maxdepth 1 -type f -name 'prompt-*.mp4' \
        | wc -l
}

if [[ "${RVM_FORCE_CALIBRATION_BANK:-0}" == "1" ]] ||
   (( $(video_count) < CALIBRATION_VIDEOS )); then
    rm -rf "${CALIBRATION_ROOT}"
    run_rvm_training \
        "${CALIBRATION_CONFIG}" \
        --method.validation.num_prompts "${CALIBRATION_VIDEOS}" \
        --method.validation.data_path "${RVM_TRAIN_DATA}" \
        --training.checkpoint.output_dir "${CALIBRATION_ROOT}" \
        --training.tracker.run_name "rvm-h3-calibration-bank-${CALIBRATION_VIDEOS}"
fi

FOUND_VIDEOS="$(video_count)"
if (( FOUND_VIDEOS < CALIBRATION_VIDEOS )); then
    echo "Calibration bank is incomplete: expected ${CALIBRATION_VIDEOS}, found ${FOUND_VIDEOS}." >&2
    exit 1
fi

calibration_args=(
    --video-dir "${CALIBRATION_VIDEO_DIR}"
    --prompt-file "${RVM_PROMPT_DIR}/train_h3.txt"
    --output "${MJ_VIDEO_CALIBRATION_PATH}"
    --device "${RVM_CALIBRATION_DEVICE:-cuda}"
    --batch-size "${RVM_CALIBRATION_BATCH_SIZE:-1}"
    --max-videos "${CALIBRATION_VIDEOS}"
)
if [[ -n "${RVM_CONSTANT_SCALE_FALLBACK:-}" ]]; then
    calibration_args+=(
        --constant-scale-fallback "${RVM_CONSTANT_SCALE_FALLBACK}"
    )
fi
python examples/train/rvm_h3/calibrate_reward_profile.py \
    "${calibration_args[@]}"

python - "${MJ_VIDEO_CALIBRATION_PATH}" "${CALIBRATION_VIDEOS}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
required = {
    "videoalign_ta",
    "mjvideo_cc",
    "mjvideo_fineness",
    "dynamic_tracking",
}
components = payload.get("components", {})
if set(components) != required:
    raise RuntimeError(
        f"Calibration components mismatch: {sorted(components)}"
    )
for name, entry in components.items():
    if int(entry["count"]) != expected:
        raise RuntimeError(
            f"{name} calibration count is {entry['count']}, expected {expected}"
        )
    if float(entry["scale"]) <= 0:
        raise RuntimeError(f"{name} has non-positive calibration scale")
provenance = payload.get("provenance", {})
if not str(provenance.get("prompt_file", "")).endswith("train_h3.txt"):
    raise RuntimeError(
        "Calibration must use the training prompt split, not held-out eval prompts"
    )
print(
    {
        "calibration": str(path),
        "profile": payload.get("profile"),
        "videos": expected,
        "components": sorted(components),
        "prompt_split": "train",
    }
)
PY

echo "Physion/MJ-VIDEO calibration ready: ${MJ_VIDEO_CALIBRATION_PATH}"
