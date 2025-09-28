#!/usr/bin/env bash
set -euo pipefail

############################################
# Single node, 6 GPUs
############################################
NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5


export WANDB_MODE="online"
export WANDB_BASE_URL="https://api.wandb.ai"
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN   # or TORCH_SDPA
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR=/tmp/triton_cache

echo "NUM_GPUS:    $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

############################################
# Configs
############################################
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR=data/Wan-Syn_77x448x832_600k/train
VALIDATION_DATASET_FILE=data/Wan-Syn_77x448x832_600k/validation_6.json
OUTPUT_DIR="checkpoints/wan_t2v_finetune"

# Training arguments
training_args=(
  --tracker_project_name wan_t2v_distill_dmd
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --num_frames 81
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments (adjusted to 6 GPUs)
parallel_args=(
  --num_gpus 6
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 6
)

# Model arguments
model_args=(
  --model_path "$MODEL_PATH"
  --pretrained_model_name_or_path "$MODEL_PATH"
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "3"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 2000
  --weight_only_checkpointing_steps 1000
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 8
  --seed 1000
)

# DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,757,522'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3.5
)

############################################
# Launch (no rendezvous flags needed)
############################################
set -x
torchrun \
  --nproc_per_node="$NUM_GPUS" \
  fastvideo/training/wan_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"
