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

# Training arguments
training_args=(
  --tracker_project_name "cosmos2_5_t2w_finetune"
  --output_dir "checkpoints/cosmos2_5_t2w_finetune"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 8
  --num_latent_t 20
  --num_height 704
  --num_width 1280
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size $NUM_GPUS
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path $DATA_DIR
  --dataloader_num_workers 1
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 500
  --validation_sampling_steps "35"
  --validation_guidance_scale "7.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 5e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 35
  --ema_start_step 0
  # --resume_from_checkpoint "checkpoints/cosmos2_5_t2w_finetune/checkpoint-1000"
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/cosmos2_5_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
