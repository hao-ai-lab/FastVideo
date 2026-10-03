#!/bin/bash
# LoRA fine-tuning for Cosmos 2.5 text-to-world (T2W)
#
# LoRA targets: attn1/attn2 q,k,v,out projections and MLP layers.
# The lora_param_names_mapping in Cosmos25ArchConfig controls which layers
# receive LoRA adapters. Excluded layers are listed in exclude_lora_layers.
#
# Prerequisites: same as finetune_t2w.sh

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="KyleShao/Cosmos-Predict2.5-2B-Diffusers"
DATA_DIR="data/cosmos2_5_processed_t2w/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=1

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
  --master_port 29501 \
    fastvideo/training/cosmos2_5_training_pipeline.py \
    --config examples/training/finetune/cosmos2_5/finetune_t2w_lora.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.sp_size "$NUM_GPUS" \
    --engine.parallelism.tp_size "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
