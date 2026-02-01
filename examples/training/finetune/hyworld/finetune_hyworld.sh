#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# HYWorld Training Script for FastVideo
# Adapted from HY-WorldPlay trainer

set -euo pipefail

# =============================================================================
# Environment
# =============================================================================
# export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
# Memory optimization: helps avoid CUDA OOM fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH=../models/HY-WorldPlay-AR-Diffusers
TRAIN_JSON=hyw/data/sythball_v1_125f_8chunk_4dirback_modelinput/sythball_v1_125f_8chunk_4dirback_train_for_hyworld.json
OUT_DIR=hyw/outputs/hyworld_sythball_125f_8chunk_4dirback_small

# Expand ~ to $HOME
MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
TRAIN_JSON="${TRAIN_JSON/#\~/$HOME}"
OUT_DIR="${OUT_DIR/#\~/$HOME}"

# =============================================================================
# GPU config
# =============================================================================
NUM_GPUS=4
SP_SIZE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29612

# =============================================================================
# WandB
# =============================================================================
export WANDB_API_KEY=XXXX
export WANDB_ENTITY=XXXX
WANDB_PROJECT=XXXX
WANDB_RUN_NAME=XXXX

NUM_FRAMES=125
NUM_LATENT_T=32
NUM_HEIGHT=256
NUM_WIDTH=256
MAX_TRAIN_STEPS=2000
LOG_STEPS=25

# =============================================================================
# Training arguments
# =============================================================================
training_args=(
  --json-path "${TRAIN_JSON}"
  --data-path "${TRAIN_JSON}"
  --causal
  --action
  --i2v-rate 0.2
  --train-time-shift 3.0
  --window-frames 16
  --output-dir "${OUT_DIR}"
  --max-train-steps ${MAX_TRAIN_STEPS}
  --train-batch-size 1
  --train-sp-batch-size 1
  --gradient-accumulation-steps 1
  --num-latent-t ${NUM_LATENT_T}
  --num-height ${NUM_HEIGHT}
  --num-width ${NUM_WIDTH}
  --num-frames ${NUM_FRAMES}
  --seed 3208
  --train-video-log-steps ${LOG_STEPS}
  --train-video-log-max-samples 1
  --train-video-log-fps 25
  --tracker-project-name "${WANDB_PROJECT}"
  --wandb-run-name "${WANDB_RUN_NAME}"
)

# =============================================================================
# Parallel arguments
# =============================================================================
parallel_args=(
  --num-gpus ${NUM_GPUS}
  --sp-size ${SP_SIZE}
  --tp-size 1
  --hsdp-replicate-dim 1
  --hsdp-shard-dim ${NUM_GPUS}
)

# =============================================================================
# Model arguments
# =============================================================================
model_args=(
  --model-path "${MODEL_PATH}"
  --pretrained-model-name-or-path "${MODEL_PATH}"
  --mode finetuning
  --workload-type i2v
)

# =============================================================================
# Dataset arguments
# =============================================================================
dataset_args=(
  --dataloader-num-workers 0
)

# =============================================================================
# Optimizer arguments
# =============================================================================
optimizer_args=(
  --learning-rate 2e-5
  --mixed-precision bf16
  --training-state-checkpointing-steps 500
  --weight-decay 1e-4
  --max-grad-norm 1.0
  --lr-scheduler "constant"
  --lr-warmup-steps 0
)

# =============================================================================
# Miscellaneous arguments
# =============================================================================
miscellaneous_args=(
  --inference-mode False
  --checkpoints-total-limit 3
  --training-cfg-rate 0.0
  --not-apply-cfg-solver
  --dit-precision fp32
  --num-euler-timesteps 50
  --ema-start-step 0
)

# =============================================================================
# Validation
# =============================================================================
if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${TRAIN_JSON}" ]; then
  echo "ERROR: TRAIN_JSON not found: ${TRAIN_JSON}" >&2
  exit 1
fi

# =============================================================================
# Print config
# =============================================================================
echo "=============================================="
echo "HYWorld Training (FastVideo)"
echo "=============================================="
echo "Model:        ${MODEL_PATH}"
echo "Train JSON:   ${TRAIN_JSON}"
echo "Output:       ${OUT_DIR}"
echo "GPUs:         ${NUM_GPUS} (SP: ${SP_SIZE})"
echo "=============================================="

# =============================================================================
# Run training
# =============================================================================
cd "${REPO_ROOT}"

torchrun \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes 1 \
  -m fastvideo.training.hyworld_training_pipeline \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${optimizer_args[@]}" \
  "${miscellaneous_args[@]}"

echo "Training complete!"
