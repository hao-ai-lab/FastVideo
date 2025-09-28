#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DATA_DIR="data/crush-smol_processed_i2v/combined_parquet_dataset/"
VALIDATION_DIR="data/crush-smol_processed_i2v/validation_parquet_dataset/"
NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name "wan_i2v_finetune"
  --output_dir "checkpoints/wan_i2v_finetune"
  --max_train_steps 2000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 8
  --num_height 480
  --num_width 832
  --num_frames 77
  --lora_rank 32
  --lora_training True
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size $NUM_GPUS
  --tp_size $NUM_GPUS
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
  --data_path "$DATA_DIR"
  --dataloader_num_workers 1
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_preprocessed_path "$VALIDATION_DIR"
  --validation_steps 100
  --validation_sampling_steps "40"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
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
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_i2v_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
