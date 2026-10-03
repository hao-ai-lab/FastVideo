#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"
DATA_DIR="footsies-dataset/preprocessed/combined_parquet_dataset"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/matrixgame2_ar_diffusion_pipeline.py \
    --config examples/training/finetune/MatrixGame2.0/finetune_df.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
