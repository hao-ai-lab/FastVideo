#!/bin/bash
set -euo pipefail

# -----------------------------
# Environment
# -----------------------------
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

# Strongly recommended on shared nodes (prevents ranks from trying to use non-visible GPUs)
# If your scheduler already sets CUDA_VISIBLE_DEVICES correctly, you can comment this out.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional: helps fragmentation sometimes
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------
# Paths / Config
# -----------------------------
MODEL_PATH="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
DATA_DIR="data/crush-smol_processed_t2v_hunyuan15/combined_parquet_dataset"
VALIDATION_DATASET_FILE="examples/training/finetune/hunyuan15_t2v/crush_smol/validation.json"

NUM_GPUS=4

# Must satisfy sp_size * tp_size == num_gpus
SP_SIZE=$NUM_GPUS 
TP_SIZE=1

# Must satisfy: hsdp_shard_dim <= num_gpus AND num_gpus % hsdp_shard_dim == 0
HSDP_REPLICATE_DIM=1 
HSDP_SHARD_DIM=$NUM_GPUS 

if (( SP_SIZE * TP_SIZE != NUM_GPUS )); then
  echo "ERROR: sp_size*tp_size must equal num_gpus. Got ${SP_SIZE}*${TP_SIZE} != ${NUM_GPUS}"
  exit 1
fi
if (( NUM_GPUS % HSDP_SHARD_DIM != 0 )); then
  echo "ERROR: num_gpus must be divisible by hsdp_shard_dim. Got ${NUM_GPUS} % ${HSDP_SHARD_DIM} != 0"
  exit 1
fi

# -----------------------------
# Args (USE UNDERSCORES: matches TrainingArgs/FastVideoArgs)
# -----------------------------
parallel_args=(
  --num_gpus "${NUM_GPUS}"
  --sp_size "${SP_SIZE}"
  --tp_size "${TP_SIZE}"
  --hsdp_replicate_dim "${HSDP_REPLICATE_DIM}"
  --hsdp_shard_dim "${HSDP_SHARD_DIM}"
)

model_args=(
  --model_path "${MODEL_PATH}"
  --pretrained_model_name_or_path "${MODEL_PATH}"
)

dataset_args=(
  --data_path "${DATA_DIR}"
  --dataloader_num_workers 1
)

training_args=(
  --mode finetuning
  --workload_type t2v
  --inference_mode False

  --tracker_project_name "hunyuan15_t2v_finetune"
  --output_dir "${DATA_DIR}/outputs_hunyuan15/crushsmol_finetune"

  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 8

  --num_latent_t 20
  --num_height 480
  --num_width 832
  --num_frames 77

  --enable_gradient_checkpointing_type full
)

validation_args=(
  --log_validation
  --validation_dataset_file "${VALIDATION_DATASET_FILE}"
  --validation_steps "100"
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 4e-6
  --mixed_precision bf16
  --weight_only_checkpointing_steps 2000
  --training_state_checkpointing_steps 2000
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

misc_args=(
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision fp32
  --num_euler_timesteps 50
  --ema_start_step 0
)

# -----------------------------
# Launch
# -----------------------------
torchrun \
  --nnodes 1 \
  --nproc_per_node "${NUM_GPUS}" \
  fastvideo/training/hunyuan15_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${misc_args[@]}"
