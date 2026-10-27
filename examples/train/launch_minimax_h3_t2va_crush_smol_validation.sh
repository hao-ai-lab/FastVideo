#!/bin/bash
# Submit the 400-step MiniMax H3 single-sample SFT experiment on eight H200 nodes.

set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-add-h3-sft
ENV_FILE=/mnt/weka/shrd/wm/junda/fv-hub/.env
CONFIG_FILE="${WORKTREE_ROOT}/examples/train/configs/overfit_minimax_h3_t2va.yaml"
PROMPT_FILE="${WORKTREE_ROOT}/examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/validation.json"
DATASET_MANIFEST="${WORKTREE_ROOT}/data/crush-smol/videos2caption.json"
TRAINING_VIDEO="${WORKTREE_ROOT}/data/crush-smol/videos/1gGQy4nxyUo-Scene-016.mp4"
TRAINING_PARQUET="${WORKTREE_ROOT}/data/crush-smol_h3_t2va_single_sample_preprocessed/data_00000.parquet"
MODEL_INDEX="${WORKTREE_ROOT}/data/models/MiniMax-H3/model_index.json"
RESULT_DIR="${WORKTREE_ROOT}/runs/minimax_h3_t2va_crush_smol_single_sample_overfit"
SNAPSHOT_DIR="${RESULT_DIR}/snapshot"
VENV_BIN=/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin

[[ -f "${ENV_FILE}" ]] || { echo "Missing environment file: ${ENV_FILE}" >&2; exit 1; }
[[ -f "${CONFIG_FILE}" ]] || { echo "Missing experiment config: ${CONFIG_FILE}" >&2; exit 1; }
[[ -f "${PROMPT_FILE}" ]] || { echo "Missing validation prompts: ${PROMPT_FILE}" >&2; exit 1; }
[[ -f "${DATASET_MANIFEST}" ]] || { echo "Missing Crush-Smol manifest: ${DATASET_MANIFEST}" >&2; exit 1; }
[[ -f "${TRAINING_VIDEO}" ]] || { echo "Missing Crush-Smol training video: ${TRAINING_VIDEO}" >&2; exit 1; }
[[ -f "${TRAINING_PARQUET}" ]] || { echo "Missing training parquet: ${TRAINING_PARQUET}" >&2; exit 1; }
[[ -f "${MODEL_INDEX}" ]] || { echo "Missing MiniMax H3 checkpoint: ${MODEL_INDEX}" >&2; exit 1; }
[[ -x "${VENV_BIN}/python" ]] || { echo "Missing validation environment: ${VENV_BIN}" >&2; exit 1; }
[[ -x "${VENV_BIN}/torchrun" ]] || { echo "Missing training environment: ${VENV_BIN}" >&2; exit 1; }

set -a
source "${ENV_FILE}"
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY is required in ${ENV_FILE}}"

export WANDB_MODE=online
export PYTHONPATH="${WORKTREE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${VENV_BIN}:${PATH}"
export PARTITION=main
export NUM_GPUS=8
export CPUS_PER_TASK=128
export MEM=1440G
export JOB_NAME=minimax_h3_t2va_crush_smol_single_sample_overfit
export OUTPUT_DIR="${RESULT_DIR}/slurm"
export SBATCH_CONSTRAINT=nvidia_h200
export SBATCH_TIMELIMIT=72:00:00

cd "${WORKTREE_ROOT}"
"${VENV_BIN}/python" -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit \
    --validate-only

mkdir -p "${OUTPUT_DIR}" "${SNAPSHOT_DIR}"

cp "${CONFIG_FILE}" "${SNAPSHOT_DIR}/experiment-config.yaml"
cp "${PROMPT_FILE}" "${SNAPSHOT_DIR}/validation.json"
cp "${WORKTREE_ROOT}/new-sft-plan-h3.md" "${SNAPSHOT_DIR}/implementation-plan.md"
git rev-parse HEAD > "${SNAPSHOT_DIR}/source-revision.txt"
git status --short > "${SNAPSHOT_DIR}/worktree-status.txt"
git diff --binary > "${SNAPSHOT_DIR}/tracked-source.patch"
git ls-files --modified --others --exclude-standard \
    | awk '$0 !~ /^(archived|data|runs)\//' \
    > "${SNAPSHOT_DIR}/source-files.txt"
tar -czf "${SNAPSHOT_DIR}/source-files.tar.gz" \
    -T "${SNAPSHOT_DIR}/source-files.txt"
printf '%s\n' 'bash examples/train/launch_minimax_h3_t2va_crush_smol_validation.sh' \
    > "${SNAPSHOT_DIR}/launch-command.txt"

echo "Experiment:   ${JOB_NAME}"
echo "Allocation:   8 nodes, 64 NVIDIA H200 GPUs, 1024 CPUs, 11520 GiB host memory"
echo "Result dir:   ${RESULT_DIR}"
echo "W&B project:  fastvideo_minimax_h3"
echo "W&B run:      minimax_h3_t2va_crush_smol_single_sample_overfit"

bash examples/train/run_slurm.sh "${CONFIG_FILE}" 8
