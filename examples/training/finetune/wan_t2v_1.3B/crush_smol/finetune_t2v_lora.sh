#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=4,5


# Training arguments
training_args=(
  --tracker_project_name "wan_t2v_finetune"
  --output_dir "checkpoints/wan_t2v_finetune_lora"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 8
  --num_latent_t 20
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
  --data_path $DATA_DIR
  --dataloader_num_workers 1
)

# Validation arguments
validation_args=(
  --log_validation 
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 200
  --validation_sampling_steps "50" 
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 5e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 400
  --training_state_checkpointing_steps 400
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
  --resume_from_checkpoint "checkpoints/wan_t2v_finetune_lora/checkpoint-160"
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
  --master_port 29501 \
    fastvideo/training/wan_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
