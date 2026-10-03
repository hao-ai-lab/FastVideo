#!/bin/bash

# Basic Info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export MASTER_PORT=29501
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# Configs
NUM_GPUS=1
MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DATA_DIR="data/crush-smol_processed_ti2v/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="examples/distill/Wan2.2-TI2V-5B-Diffusers/crush_smol/validation.json"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

torchrun \
--nnodes 1 \
--nproc_per_node $NUM_GPUS \
--master_port $MASTER_PORT \
    fastvideo/training/wan_distillation_pipeline.py \
    --config examples/distill/Wan2.2-TI2V-5B-Diffusers/crush_smol/distill_dmd_VSA_t2v_5B_lora.yaml \
    --model_path "$MODEL_PATH" \
    --training.distillation.real_score_model_path "$REAL_SCORE_MODEL_PATH" \
    --training.distillation.fake_score_model_path "$FAKE_SCORE_MODEL_PATH" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
