#!/bin/bash
# Wan 2.1 GRPO training on 8 GPUs

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY="your-api-key"
export WANDB_PROJECT="your-project-name"

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
RL_DATASET_DIR="data/ocr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_DATASET_FILE="$SCRIPT_DIR/validation.json"
NUM_GPUS=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

training_args=(
  --tracker_project_name "$WANDB_PROJECT"
  --output_dir "checkpoints/${WANDB_PROJECT}_4gpu"
  --max_train_steps 3000
  --train_batch_size 8
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 240
  --num_width 416
  --num_frames 33
  --lora_rank 32
  --lora_training True
)

parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1 
  --tp_size 1 
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

model_args=( --model_path $MODEL_PATH --pretrained_model_name_or_path $MODEL_PATH )
dataset_args=(
  --data_path $RL_DATASET_DIR
  --rl_dataset_path $RL_DATASET_DIR
  --rl_dataset_type "text"
  --rl_num_image_per_prompt 4
  --dataloader_num_workers 1
)
validation_args=(
  --log_validation True
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 30
  --validation_sampling_steps "20"
  --validation_guidance_scale "4.5"
)
optimizer_args=(
  --learning_rate 1e-4
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 60
  --training_state_checkpointing_steps 60
  --weight_decay 1e-4
  --max_grad_norm 1.0
)
rl_args=(
  --inference_mode False
  --rl_mode True
  --rl_algorithm "grpo"
  --rl_kl_beta 0.004
  --rl_policy_clip_range 0.001
  --rl_kl_reward 0.0
  --rl_global_std False
  --rl_per_prompt_stat_tracking True
  --rl_num_batches_per_step 2
  --rl_warmup_steps 0
  --reward-models "{\"paddle_ocr\": 1.0}"
)
cfg_args=( 
  --guidance_scale 4.5 
)
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 67
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
  --enable-gradient-checkpointing-type "full"
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
  --master_port 26222 \
  "fastvideo/training/wan_rl_training_pipeline.py" \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${rl_args[@]}" \
  "${cfg_args[@]}" \
  "${miscellaneous_args[@]}"
