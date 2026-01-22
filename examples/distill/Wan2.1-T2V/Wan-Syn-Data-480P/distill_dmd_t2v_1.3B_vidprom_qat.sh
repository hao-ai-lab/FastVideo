#!/usr/bin/env bash
set -euo pipefail

############################################
# Single node, 6 GPUs
############################################
NUM_GPUS=1
export WANDB_API_KEY=2f25ad37933894dbf0966c838c0b8494987f9f2f
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
DATA_DIR=data/vidprom_16k_umt5_text_embed
VALIDATION_DATASET_FILE="examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json"
OUTPUT_DIR="checkpoints/wan_1.3B_t2v_distill_dmd_qat_vidprom"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Training arguments
training_args=(
  --tracker_project_name wan_t2v_distill_dmd_qat
  --output_dir $OUTPUT_DIR
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --num_frames 81
  --enable_gradient_checkpointing_type "full"
  --generator_4bit_attn True
)

# Parallel arguments (adjusted to 6 GPUs)
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
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
  --validation_steps 200
  --validation_sampling_steps "3"
  --validation_guidance_scale "6.0" # not used for dmd inference
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
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
  --simulate_generator_forward
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