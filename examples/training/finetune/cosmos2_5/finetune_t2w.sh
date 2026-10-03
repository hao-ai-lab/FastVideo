#!/bin/bash
# Full fine-tuning for Cosmos 2.5 text-to-world (T2W)
#
# Prerequisites:
#   1. Pre-process your dataset with the parquet pipeline (same format as Wan T2V).
#      Text embeddings must be pre-computed with the Reason1 (Qwen2.5-VL)
#      encoder using embedding_concat_strategy="full_concat" (100352-dim output).
#      Latents must be pre-encoded with the Cosmos25WanVAEWrapper (normalisation
#      is applied inside the encoder, so stored latents are already normalised).
#   2. Populate validation.json in this directory.
#
# Resolution guide (reduce to fit GPU memory):
#   Native  : 704x1280, num_latent_t=20  (~72B tokens / sample)
#   Reduced : 480x832,  num_latent_t=20  (~32B tokens / sample)

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="KyleShao/Cosmos-Predict2.5-2B-Diffusers"
DATA_DIR="data/cosmos2_5_processed_t2w/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=4

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/cosmos2_5_training_pipeline.py \
    --config examples/training/finetune/cosmos2_5/finetune_t2w.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.sp_size "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
