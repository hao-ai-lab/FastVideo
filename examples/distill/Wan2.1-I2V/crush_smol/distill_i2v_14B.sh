#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

MODEL_PATH="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DATA_DIR="data/crush-smol_processed_wan21_i2v_14b/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="examples/distill/Wan2.1-I2V/crush_smol/validation.json"
NUM_GPUS=8
export WANDB_API_KEY='8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde'
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name "wan_i2v_distill"
  --output_dir "checkpoints/wan_i2v_distill"
  --wandb_run_name "wan21_i2v_14b_ode_3steps"
  --max_train_steps 2000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 8
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 8
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
  # --log_visualization
  --log_visualization
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 50
  --validation_sampling_steps "3"
  --validation_guidance_scale "1.0" # not used for dmd inference
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 4e-6
  --lr_scheduler "constant"
  --fake_score_learning_rate 8e-7
  --fake_score_lr_scheduler "constant"
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 2000
  --weight_only_checkpointing_steps 2000
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
  # --enable_gradient_checkpointing_type "full"
)

dmd_args=(
  --dmd_denoising_steps '1000,757,522'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --simulate_generator_forward
  --real_score_guidance_scale 3.5
  --VSA_sparsity 0.8
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_i2v_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"