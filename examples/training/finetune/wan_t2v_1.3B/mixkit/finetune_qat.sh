#!/bin/bash
# QAD recipe — quantization-aware finetune of Wan2.1-T2V-1.3B with fake-quant
# (Attn-QAT) attention.
#
# The 4-bit attention path is selected by env var:
# FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN keeps the fake-quantized Triton
# forward and backward, so the DiT learns to absorb FP4 attention error instead
# of fighting it. SM100 selects the optimized kernels; RTX 5090 keeps the
# previous Triton implementation.
#
# Data: run examples/datasets/mixkit/download_dataset.sh first (preprocessed Parquet).
#
# Verified end-to-end on Blackwell (GB200/sm_100): the ATTN_QAT_TRAIN backend is
# selected (not a fallback), forward+backward run, loss/grad are healthy, and
# validation generates videos.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/fastvideo-kernel/python${PYTHONPATH:+:${PYTHONPATH}}"
export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN   # <-- enables Attn-QAT training
export FASTVIDEO_ATTN_QAT_FWD_EXACT_M=${FASTVIDEO_ATTN_QAT_FWD_EXACT_M:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR=${1:-"data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset/"}
VALIDATION_FILE="$(dirname "$0")/../crush_smol/validation.json"
NUM_GPUS=${NUM_GPUS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-4000}
VALIDATION_SAMPLING_STEPS=${VALIDATION_SAMPLING_STEPS:-50}

torchrun --nnodes 1 --nproc_per_node "${NUM_GPUS}" \
    fastvideo/training/wan_training_pipeline.py \
    --config examples/training/finetune/wan_t2v_1.3B/mixkit/finetune_qat.yaml \
    --model_path "${MODEL_PATH}" \
    --engine.num_gpus "${NUM_GPUS}" \
    --engine.parallelism.sp_size "${NUM_GPUS}" \
    --engine.parallelism.hsdp_shard_dim "${NUM_GPUS}" \
    --training.data.data_path "${DATA_DIR}" \
    --training.loop.max_train_steps "${MAX_TRAIN_STEPS}" \
    --training.validation.dataset_file "${VALIDATION_FILE}" \
    --training.validation.sampling_steps "[${VALIDATION_SAMPLING_STEPS}]"
