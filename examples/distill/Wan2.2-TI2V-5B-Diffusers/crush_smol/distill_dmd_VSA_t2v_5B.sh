#!/bin/bash

# Basic Info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# Configs
NUM_GPUS=1
MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers" 
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DATA_DIR="data/crush-smol_processed_ti2v/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="examples/distill/Wan2.2-TI2V-5B-Diffusers/crush_smol/validation.json"
OUTPUT_DIR="checkpoints/wan_t2v_finetune"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name wan_t2v_distill_dmd_VSA
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 31
  --num_height 704
  --num_width 1280
  --num_frames 121
  --enable_gradient_checkpointing_type "full"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
)

# Parallel arguments
parallel_args=(
  --num_gpus 1
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 1
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
  --VSA_sparsity 0.8
)

torchrun \
--nnodes 1 \
--nproc_per_node $NUM_GPUS \
--master_port $MASTER_PORT \
    fastvideo/training/wan_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"
