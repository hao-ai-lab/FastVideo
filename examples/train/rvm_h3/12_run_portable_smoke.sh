#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Portable one-/four-GPU smoke orchestration. This is the source of truth used
# by Modal, but it also runs unchanged on any Docker host or cloud VM.
MODE="${RVM_SMOKE_MODE:-all}"
GPUS="${RVM_SMOKE_GPUS:-1}"
RUN_NAME="${RVM_SMOKE_RUN_NAME:-auto}"
RUN_ROOT="${RVM_SMOKE_RUN_ROOT:-${REPO_ROOT}/outputs/rvm_h3/portable_smoke}"
MAX_STEPS="${RVM_SMOKE_MAX_STEPS:-0}"
EVAL_PROMPTS="${RVM_SMOKE_EVAL_PROMPTS:-8}"
MAX_TRAIN_PROMPTS="${RVM_SMOKE_MAX_TRAIN_PROMPTS:-64}"
SMOKE_PROMPTS="${RVM_SMOKE_PROMPTS:-16}"
LEARNING_RATE="${RVM_SMOKE_LEARNING_RATE:-0}"
CONFIG="${RVM_SMOKE_CONFIG:-}"
FORCE_PREPARE="${RVM_SMOKE_FORCE_PREPARE:-0}"
RESUME="${RVM_SMOKE_RESUME:-0}"
SKIP_PREFLIGHT="${RVM_SMOKE_SKIP_PREFLIGHT:-0}"
SKIP_INSTALL="${RVM_SMOKE_SKIP_INSTALL:-0}"

case "${MODE}" in
    prepare|preflight|smoke|pilot|all) ;;
    *)
        echo "RVM_SMOKE_MODE must be prepare, preflight, smoke, pilot, or all." >&2
        exit 2
        ;;
esac
if [[ "${GPUS}" != "1" && "${GPUS}" != "4" ]]; then
    echo "RVM_SMOKE_GPUS must be 1 or 4." >&2
    exit 2
fi
for value in "${EVAL_PROMPTS}" "${MAX_TRAIN_PROMPTS}" "${SMOKE_PROMPTS}"; do
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "Prompt counts must be positive integers; got ${value}." >&2
        exit 2
    fi
done
if (( EVAL_PROMPTS > 100 )); then
    echo "RVM_SMOKE_EVAL_PROMPTS must be <= 100." >&2
    exit 2
fi
if ! [[ "${MAX_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "RVM_SMOKE_MAX_STEPS must be a nonnegative integer." >&2
    exit 2
fi

if [[ -z "${CONFIG}" ]]; then
    if [[ "${GPUS}" == "1" ]]; then
        CONFIG="examples/train/configs/rl/minimax_h3/rvm_h3_modal_1gpu.yaml"
    else
        CONFIG="examples/train/configs/rl/minimax_h3/rvm_h3_modal_4gpu.yaml"
    fi
fi
if (( MAX_STEPS == 0 )); then
    if [[ "${GPUS}" == "1" ]]; then
        MAX_STEPS=1
    else
        MAX_STEPS=10
    fi
fi
if [[ "${RUN_NAME}" == "auto" || -z "${RUN_NAME}" ]]; then
    RUN_NAME="h3-rvm-${GPUS}gpu-$(date -u +%Y%m%d-%H%M%S)"
fi
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "${RUN_NAME}" == "." || "${RUN_NAME}" == ".." ]]; then
    echo "Unsafe RVM_SMOKE_RUN_NAME: ${RUN_NAME}" >&2
    exit 2
fi

RUN_DIR="${RUN_ROOT%/}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/training"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
export LOG_DIR
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
export WANDB_MODE="${WANDB_MODE:-$([[ -n "${WANDB_API_KEY:-}" ]] && echo online || echo offline)}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(printf '%s' "${RUN_NAME}" | sha1sum | cut -c1-16)}"

STATUS="failed"
write_manifest() {
    local exit_code="$1"
    RUN_STATUS="${STATUS}" RUN_EXIT_CODE="${exit_code}" python - "${RUN_DIR}/run_manifest.json" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

path = Path(sys.argv[1])
run_dir = path.parent
output_dir = run_dir / "training"


def capture(*command: str) -> str:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


checkpoints = sorted(
    item.name
    for item in output_dir.glob("checkpoint-*")
    if (item / "dcp" / ".metadata").is_file()
)
validations = sorted(
    item.name
    for item in (output_dir / "validation").glob("step-*")
    if item.is_dir()
)
payload = {
    "status": os.environ["RUN_STATUS"],
    "exit_code": int(os.environ["RUN_EXIT_CODE"]),
    "git_head": capture("git", "rev-parse", "HEAD"),
    "git_tree": capture("git", "rev-parse", "HEAD^{tree}"),
    "gpu": capture(
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader",
    ),
    "mode": os.environ["RVM_SMOKE_MODE"],
    "gpus": int(os.environ["RVM_SMOKE_GPUS"]),
    "config": os.environ["RVM_SMOKE_CONFIG_EFFECTIVE"],
    "max_steps": int(os.environ["RVM_SMOKE_MAX_STEPS_EFFECTIVE"]),
    "eval_prompts": int(os.environ["RVM_SMOKE_EVAL_PROMPTS"]),
    "train_prompt_limit": int(os.environ["RVM_SMOKE_MAX_TRAIN_PROMPTS"]),
    "output_dir": str(output_dir),
    "checkpoints": checkpoints,
    "validation_steps": validations,
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}
trap 'code=$?; write_manifest "${code}"; exit "${code}"' EXIT

export RVM_SMOKE_MODE="${MODE}"
export RVM_SMOKE_GPUS="${GPUS}"
export RVM_SMOKE_CONFIG_EFFECTIVE="${CONFIG}"
export RVM_SMOKE_MAX_STEPS_EFFECTIVE="${MAX_STEPS}"
export RVM_SMOKE_EVAL_PROMPTS="${EVAL_PROMPTS}"
export RVM_SMOKE_MAX_TRAIN_PROMPTS="${MAX_TRAIN_PROMPTS}"

python examples/train/rvm_h3/verify_clean_source.py \
    --output "${RUN_DIR}/source_manifest.json"

activate_rvm_env
if [[ "${SKIP_INSTALL}" != "1" ]]; then
    bash examples/train/rvm_h3/00_install_current_env.sh
fi

# Configs use artifacts/rvm_h3. Keep the normal local directory untouched;
# only create a symlink when a cloud service supplies a different persistent
# artifact root.
ARTIFACT_LINK="${REPO_ROOT}/artifacts/rvm_h3"
if [[ "${RVM_ARTIFACT_ROOT}" != "${ARTIFACT_LINK}" ]]; then
    mkdir -p "${REPO_ROOT}/artifacts"
    rm -rf "${ARTIFACT_LINK}"
    ln -s "${RVM_ARTIFACT_ROOT}" "${ARTIFACT_LINK}"
else
    mkdir -p "${RVM_ARTIFACT_ROOT}"
fi

assets_ready() {
    [[ -d "${FASTH3_MODEL_DIR}/transformer" ]] &&
    [[ -d "${FASTH3_MODEL_DIR}/vae" ]] &&
    [[ -d "${FASTH3_MODEL_DIR}/text_encoder" ]] &&
    [[ -f "${VIDEOALIGN_RUNTIME_PATH}/inference.py" ]] &&
    [[ -d "${VIDEOALIGN_CHECKPOINT_PATH}" ]]
}
if [[ "${FORCE_PREPARE}" == "1" ]] || ! assets_ready; then
    bash examples/train/rvm_h3/01_download_models.sh
fi

parquet_rows() {
    python - "$1" <<'PY'
from pathlib import Path
import sys

import pyarrow.parquet as pq

root = Path(sys.argv[1])
print(
    sum(
        int(pq.ParquetFile(path).metadata.num_rows)
        for path in root.rglob("*.parquet")
    )
    if root.is_dir()
    else 0
)
PY
}

train_rows="$(parquet_rows "${RVM_TRAIN_DATA}")"
eval_rows="$(parquet_rows "${RVM_EVAL_DATA}")"
smoke_rows="$(parquet_rows "${RVM_SMOKE_DATA}")"
if [[ "${FORCE_PREPARE}" == "1" ]] ||
   (( train_rows < MAX_TRAIN_PROMPTS )) ||
   (( eval_rows < EVAL_PROMPTS )) ||
   (( smoke_rows < SMOKE_PROMPTS )); then
    RVM_FORCE_PREPROCESS=1 \
    RVM_MAX_TRAIN_PROMPTS="${MAX_TRAIN_PROMPTS}" \
    RVM_EVAL_PROMPTS="${EVAL_PROMPTS}" \
    RVM_SMOKE_PROMPTS="${SMOKE_PROMPTS}" \
    RVM_PREPROCESS_GPUS="${GPUS}" \
        bash examples/train/rvm_h3/02_prepare_dataset.sh
fi

if [[ "${MODE}" == "prepare" ]]; then
    STATUS="succeeded"
    exit 0
fi

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    if ! find "${RVM_ARTIFACT_ROOT}/inference_smoke" -name '*.mp4' -print -quit 2>/dev/null | grep -q .; then
        bash examples/train/rvm_h3/03_public_inference_smoke.sh
    fi
    RVM_PREFLIGHT_REWARD_OUTPUT="${RUN_DIR}/preflight_reward_scores.json" \
        bash examples/train/rvm_h3/03_preflight_1gpu.sh
fi

if [[ "${MODE}" == "preflight" ]]; then
    STATUS="succeeded"
    exit 0
fi

DATA_PATH="${RVM_TRAIN_DATA}"
if [[ "${GPUS}" == "1" ]]; then
    DATA_PATH="${RVM_SMOKE_DATA}"
fi
CHECKPOINT_INTERVAL=$(( MAX_STEPS / 2 ))
if (( CHECKPOINT_INTERVAL < 1 )); then
    CHECKPOINT_INTERVAL=1
fi

train_args=(
    --models.student.init_from "${FASTH3_MODEL_DIR}"
    --training.model_path "${FASTH3_MODEL_DIR}"
    --training.data.data_path "${DATA_PATH}"
    --method.validation.data_path "${RVM_EVAL_DATA}"
    --method.validation.num_prompts "${EVAL_PROMPTS}"
    --method.validation.log_sample_limit "$(( EVAL_PROMPTS < 8 ? EVAL_PROMPTS : 8 ))"
    --method.validation.every_steps "${MAX_STEPS}"
    --training.loop.max_train_steps "${MAX_STEPS}"
    --training.checkpoint.output_dir "${OUTPUT_DIR}"
    --training.checkpoint.training_state_checkpointing_steps "${CHECKPOINT_INTERVAL}"
    --training.tracker.run_name "${RUN_NAME}"
)
if python -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)' "${LEARNING_RATE}"; then
    train_args+=(--training.optimizer.learning_rate "${LEARNING_RATE}")
fi
if [[ "${RESUME}" == "1" ]] &&
   find "${OUTPUT_DIR}" -path '*/dcp/.metadata' -print -quit 2>/dev/null | grep -q .; then
    train_args+=(--training.checkpoint.resume_from_checkpoint latest)
fi

export NUM_GPUS="${GPUS}"
export RVM_SP_SIZE="${GPUS}"
run_rvm_training "${CONFIG}" "${train_args[@]}"

python examples/train/rvm_h3/11_collect_results.py \
    --root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/results_index.json"

STATUS="succeeded"
