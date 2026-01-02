#!/bin/bash

# Change to FastVideo root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTVIDEO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$FASTVIDEO_ROOT"

# Add FastVideo root to PYTHONPATH so Python can find the fastvideo package
export PYTHONPATH="$FASTVIDEO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
RL_DATASET_DIR="data/ocr/"  # Path to RL prompt dataset directory (should contain train.txt and test.txt)
VALIDATION_DATASET_FILE="$SCRIPT_DIR/validation.json"
NUM_GPUS=1

# use GPU 3
export CUDA_VISIBLE_DEVICES=3


# Training arguments
training_args=(
  --tracker_project_name "wan_t2v_grpo"
  --output_dir "checkpoints/wan_t2v_grpo"
  --max_train_steps 5000
  --train_batch_size 4
  # --train_sp_batch_size 4
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
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
  # --hsdp_replicate_dim 1
  # --hsdp_shard_dim $NUM_GPUS
  --use-fsdp-inference False
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments (for RL prompt dataset)
dataset_args=(
  --data_path $RL_DATASET_DIR  # Used as fallback if rl_dataset_path not set
  --rl_dataset_path $RL_DATASET_DIR  # RL prompt dataset directory
  --rl_dataset_type "text"  # "text" or "geneval"
  --rl_num_image_per_prompt 4  # k parameter (number of samples per prompt)
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

# RL-specific arguments
rl_args=(
  --rl_mode True
  --rl_algorithm "grpo"
  --rl_kl_beta 0.004  # KL regularization coefficient
  --rl_policy_clip_range 0.2  # Policy clipping range for GRPO
  --rl_kl_reward 0.0  # KL reward coefficient (typically 0)
  --rl_global_std False  # Use per-prompt std (recommended for GRPO)
  --rl_per_prompt_stat_tracking True  # Enable per-prompt stat tracking
  --rl_warmup_steps 0  # Number of warmup steps (SFT before RL)
)

# CFG arguments
cfg_args=(
  --guidance_scale 1.0 # use guidance_scale > 1.0 to enable CFG
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0  # No CFG during training (CFG used in sampling)
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
  # --resume_from_checkpoint "checkpoints/wan_t2v_grpo/checkpoint-XXX"
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
  --master_port 29501 \
    "$FASTVIDEO_ROOT/fastvideo/training/wan_rl_training_pipeline.py" \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${rl_args[@]}" \
    "${miscellaneous_args[@]}"
