#!/usr/bin/env bash
# Train an official MMAudio v1 architecture from random DiT initialization.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ---------------------------------------------------------------------------
# Edit the training run here. No environment variables are required.
# ---------------------------------------------------------------------------
VARIANT="small_44k"

FEATURE_DIR="/mnt/lustre/vlm-kai/datasets/VGGSound/mmaudio_features_torio/train"
VALIDATION_FEATURE_DIR="/mnt/lustre/vlm-kai/datasets/VGGSound/mmaudio_features_torio/val"
EMPTY_STRING_FEATURES="${REPO_ROOT}/official_weights/mmaudio/raw/ext_weights/empty_string.pth"
EMPTY_STRING_URL="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth"

NUM_GPUS=4
PER_GPU_BATCH_SIZE=128
GRAD_ACCUM_STEPS=1
# This container has a 64 MB /dev/shm. Keep these at zero unless the job is
# launched with a larger shared-memory mount.
DATALOADER_NUM_WORKERS=0
VALIDATION_NUM_WORKERS=0

LEARNING_RATE="1.0e-4"
MAX_TRAIN_STEPS=300000
CHECKPOINT_EVERY_STEPS=10000
VALIDATION_EVERY_STEPS=5000
# 0 evaluates the complete validation split.
VALIDATION_MAX_BATCHES=0
INFERENCE_EVERY_STEPS=20000
INFERENCE_NUM_SAMPLES=16
INFERENCE_NUM_STEPS=25
INFERENCE_SAVE_VIDEO=true
INFERENCE_LOG_TO_TRACKER=true
INFERENCE_MODEL_PATH="${REPO_ROOT}/converted_weights/mmaudio/large_44k_v2"

OUTPUT_DIR="${REPO_ROOT}/outputs/mmaudio_${VARIANT}_from_scratch"
RESUME_FROM_CHECKPOINT=""
# The API key is stored by `wandb login` in the current user's credentials;
# never put it in this tracked launcher.
TRACKER="wandb"
TRACKER_PROJECT="fastvideo_mmaudio"

# Prefer the repository environment so the script also works from a fresh
# non-activated shell. An already activated external environment remains the
# fallback when .venv is absent.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

for required_dir in "${FEATURE_DIR}" "${VALIDATION_FEATURE_DIR}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "MMAudio feature directory does not exist: ${required_dir}" >&2
        exit 2
    fi
done

# Official TRAINING.md publishes this small standalone CLIP encoding. It is
# fixed conditioning data, not a pretrained MMAudio transformer checkpoint.
if [[ ! -f "${EMPTY_STRING_FEATURES}" ]]; then
    mkdir -p "$(dirname "${EMPTY_STRING_FEATURES}")"
    echo "Downloading official empty-string encoding to ${EMPTY_STRING_FEATURES}"
    curl --fail --location --continue-at - \
        --output "${EMPTY_STRING_FEATURES}" \
        "${EMPTY_STRING_URL}"
fi

case "${VARIANT}" in
    small_16k)
        CONFIG_NAME="small_16k_from_scratch.yaml"
        STATS_NAME="latent_statistics_16k.pt"
        INFERENCE_EVERY_STEPS=0
        INFERENCE_MODEL_PATH=""
        ;;
    small_44k|medium_44k|large_44k)
        CONFIG_NAME="${VARIANT}_from_scratch.yaml"
        STATS_NAME="latent_statistics_44k.pt"
        ;;
    large_44k_v2)
        echo "Official MMAudio training does not support large_44k_v2." >&2
        exit 2
        ;;
    *)
        echo "Unknown VARIANT=${VARIANT}" >&2
        exit 2
        ;;
esac

CONFIG="${REPO_ROOT}/examples/train/configs/fine_tuning/mmaudio/${CONFIG_NAME}"
LATENT_STATISTICS="${FEATURE_DIR}/${STATS_NAME}"

export NUM_GPUS
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
    --training.distributed.num_gpus "${NUM_GPUS}" \
    --training.distributed.hsdp_shard_dim "${NUM_GPUS}" \
    --training.checkpoint.output_dir "${OUTPUT_DIR}" \
    --training.checkpoint.training_state_checkpointing_steps "${CHECKPOINT_EVERY_STEPS}" \
    --training.checkpoint.resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" \
    --training.tracker.trackers "[${TRACKER}]" \
    --training.tracker.project_name "${TRACKER_PROJECT}" \
    --training.tracker.run_name "mmaudio_${VARIANT}_from_scratch" \
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
