#!/bin/bash
set -e -x

# Phase 1 example: run Wan DMD2 distillation via DMD2Method + WanAdapter entrypoint.
# Note: validation is currently best-effort; Phase 1 focuses on algorithm/model decoupling.

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}
export WANDB_MODE=${WANDB_MODE:-offline}
export MASTER_PORT=${MASTER_PORT:-29504}

NUM_GPUS=${NUM_GPUS:-1}

# Models
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}
# For best distillation, point TEACHER_MODEL_PATH to a stronger teacher (e.g. 14B).
# For a cheaper smoke run, set it to the same 1.3B model.
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"Wan-AI/Wan2.1-T2V-14B-Diffusers"}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}

# Data (parquet dataset folder)
DATA_DIR=${DATA_DIR:-"your_data_dir"}
VALIDATION_DATASET_FILE=${VALIDATION_DATASET_FILE:-"your_validation_dataset_file"}

OUTPUT_DIR=${OUTPUT_DIR:-"outputs/phase1_wan2.1_t2v_1.3B_dmd2_8steps"}

training_args=(
  --tracker_project_name "phase1_wan_dmd2_8steps"
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
  --simulate_generator_forward
)

parallel_args=(
  --num_gpus "$NUM_GPUS"
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim "$NUM_GPUS"
)

model_args=(
  --model_path "$STUDENT_MODEL_PATH"
  --pretrained_model_name_or_path "$STUDENT_MODEL_PATH"
  --real_score_model_path "$TEACHER_MODEL_PATH"
  --fake_score_model_path "$CRITIC_MODEL_PATH"
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 50
  --validation_sampling_steps "8"
  --validation_guidance_scale "6.0" # not used for dmd inference
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
  --fake_score_learning_rate 8e-6
  --fake_score_betas '0.0,0.999'
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 8
  --seed 1000
)

dmd_args=(
  # 8-step schedule (same as Wan2.2 self-forcing examples)
  --dmd_denoising_steps '1000,850,700,550,350,275,200,125'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3.5
)

torchrun \
--nnodes 1 \
--master_port "$MASTER_PORT" \
--nproc_per_node "$NUM_GPUS" \
    fastvideo/training/wan_distillation_v3.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"

