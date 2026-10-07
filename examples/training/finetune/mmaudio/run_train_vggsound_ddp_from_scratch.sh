#!/usr/bin/env bash
# Train MMAudio small_44k with the official full-replica DDP strategy.
# This launcher is independent from run_train_vggsound_from_scratch.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ---------------------------------------------------------------------------
# Edit the DDP run here. Do not launch while another job owns the same GPUs.
# ---------------------------------------------------------------------------
FEATURE_DIR="/mnt/lustre/vlm-kai/datasets/VGGSound/mmaudio_features_torio/train"
VALIDATION_FEATURE_DIR="/mnt/lustre/vlm-kai/datasets/VGGSound/mmaudio_features_torio/val"
EMPTY_STRING_FEATURES="${REPO_ROOT}/official_weights/mmaudio/raw/ext_weights/empty_string.pth"
EMPTY_STRING_URL="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth"

NUM_GPUS=4
MASTER_PORT=29502
PER_GPU_BATCH_SIZE=128
GRAD_ACCUM_STEPS=1
# /dev/shm is currently 64 MB, so subprocess workers can exhaust shared memory.
DATALOADER_NUM_WORKERS=0
VALIDATION_NUM_WORKERS=0

LEARNING_RATE="1.0e-4"
MAX_TRAIN_STEPS=300000
COMPILE_TRAIN_FN=true
CHECKPOINT_EVERY_STEPS=10000
VALIDATION_EVERY_STEPS=5000
VALIDATION_MAX_BATCHES=0
INFERENCE_EVERY_STEPS=20000
INFERENCE_NUM_SAMPLES=16
INFERENCE_NUM_STEPS=25
INFERENCE_SAVE_VIDEO=true
INFERENCE_LOG_TO_TRACKER=true
INFERENCE_MODEL_PATH="${REPO_ROOT}/converted_weights/mmaudio/large_44k_v2"

OUTPUT_DIR="${REPO_ROOT}/outputs/mmaudio_small_44k_ddp_from_scratch"
RESUME_FROM_CHECKPOINT="${OUTPUT_DIR}/checkpoint-20000"
TRACKER="wandb"
TRACKER_PROJECT="fastvideo_mmaudio"

# These W&B settings are used only when resuming an existing checkpoint.
# For a new run, set RESUME_FROM_CHECKPOINT=""; the variables are then
# explicitly unset so training cannot accidentally append to the old page.
WANDB_RESUME_RUN_ID="vabj9rjb"
WANDB_RESUME_ENTITY="alan-wang-university-of-toronto"

CONFIG="${REPO_ROOT}/examples/train/configs/fine_tuning/mmaudio/small_44k_ddp_from_scratch.yaml"
LATENT_STATISTICS="${FEATURE_DIR}/latent_statistics_44k.pt"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

for required_dir in "${FEATURE_DIR}" "${VALIDATION_FEATURE_DIR}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "MMAudio feature directory does not exist: ${required_dir}" >&2
        exit 2
    fi
done

if [[ ! -f "${EMPTY_STRING_FEATURES}" ]]; then
    mkdir -p "$(dirname "${EMPTY_STRING_FEATURES}")"
    echo "Downloading official empty-string encoding to ${EMPTY_STRING_FEATURES}"
    curl --fail --location --continue-at - \
        --output "${EMPTY_STRING_FEATURES}" \
        "${EMPTY_STRING_URL}"
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    export WANDB_RUN_ID="${WANDB_RESUME_RUN_ID}"
    export WANDB_RESUME="must"
    export WANDB_ENTITY="${WANDB_RESUME_ENTITY}"
else
    unset WANDB_RUN_ID WANDB_RESUME WANDB_ENTITY
fi

export NUM_GPUS MASTER_PORT
cd "${REPO_ROOT}"
exec bash examples/train/run.sh "${CONFIG}" \
    --models.student.empty_string_features_path "${EMPTY_STRING_FEATURES}" \
    --models.student.latent_statistics_path "${LATENT_STATISTICS}" \
    --training.data.data_path "${FEATURE_DIR}" \
    --training.data.train_batch_size "${PER_GPU_BATCH_SIZE}" \
    --training.data.dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --training.optimizer.learning_rate "${LEARNING_RATE}" \
    --training.loop.max_train_steps "${MAX_TRAIN_STEPS}" \
    --training.loop.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --training.model.compile_train_fn "${COMPILE_TRAIN_FN}" \
    --training.distributed.num_gpus "${NUM_GPUS}" \
    --training.checkpoint.output_dir "${OUTPUT_DIR}" \
    --training.checkpoint.training_state_checkpointing_steps "${CHECKPOINT_EVERY_STEPS}" \
    --training.checkpoint.resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" \
    --training.tracker.trackers "[${TRACKER}]" \
    --training.tracker.project_name "${TRACKER_PROJECT}" \
    --training.tracker.run_name "mmaudio_small_44k_ddp_from_scratch" \
    --callbacks.validation.data_path "${VALIDATION_FEATURE_DIR}" \
    --callbacks.validation.every_steps "${VALIDATION_EVERY_STEPS}" \
    --callbacks.validation.max_batches "${VALIDATION_MAX_BATCHES}" \
    --callbacks.validation.num_data_workers "${VALIDATION_NUM_WORKERS}" \
    --callbacks.validation.inference_every_steps "${INFERENCE_EVERY_STEPS}" \
    --callbacks.validation.inference_num_samples "${INFERENCE_NUM_SAMPLES}" \
    --callbacks.validation.inference_num_steps "${INFERENCE_NUM_STEPS}" \
    --callbacks.validation.inference_save_video "${INFERENCE_SAVE_VIDEO}" \
    --callbacks.validation.inference_log_to_tracker "${INFERENCE_LOG_TO_TRACKER}" \
    --callbacks.validation.inference_model_path "${INFERENCE_MODEL_PATH}" \
    "$@"
