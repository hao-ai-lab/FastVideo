#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=4,5


torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_training_pipeline.py \
    --config examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.sp_size "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
