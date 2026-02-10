#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=""
DATA_DIR="../vizdoom/preprocessed"
VALIDATION_DATASET_FILE="examples/training/consistency_finetune/causal_wangame_ode_init/validation.json"
NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name "wangame_ode_init"
  --output_dir "checkpoints/wangame_ode_init"
  --override_transformer_cls_name "CausalWanGameActionTransformer3DModel"
  --wandb_run_name "wangame_ode_init_8gpu"
  --max_train_steps 2000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 4
  --num_latent_t 21
  --num_height 352
  --num_width 640
  --num_frames 81
  --warp_denoising_step
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
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
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR" 
  --dataloader_num_workers 1
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 6e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 100
  --training_state_checkpointing_steps 100
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
    fastvideo/training/wangame_ode_causal_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
