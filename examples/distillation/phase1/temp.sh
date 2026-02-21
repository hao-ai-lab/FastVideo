#!/bin/bash
set -e -x

# One-shot launch script for Phase1 (DMD2Method + WanAdapter) Wan DMD2 few-step distillation.
# Uses the same local Wan-Syn parquet dataset + validation json as Phase0 temp.sh.
#
# Notes:
# - By default this runs W&B in offline mode (safer for overnight runs).
#   If you want online logging:
#     export WANDB_MODE=online
#     export WANDB_API_KEY=...
# - Phase1 uses the general entrypoint:
#     fastvideo/training/distillation.py --distill_model wan --distill_method dmd2

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
export WANDB_MODE=${WANDB_MODE:-offline}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29504}

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY in env." >&2
  exit 1
fi

if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
  else
    NUM_GPUS=1
  fi
fi

# Models
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"Wan-AI/Wan2.1-T2V-14B-Diffusers"}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}

# Data (parquet dataset folder)
DATA_DIR=${DATA_DIR:-"data/Wan-Syn_77x448x832_600k"}
DEFAULT_VALIDATION_DATASET_FILE=\
"examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json"
VALIDATION_DATASET_FILE=${VALIDATION_DATASET_FILE:-"$DEFAULT_VALIDATION_DATASET_FILE"}

RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/phase1_wan2.1_dmd2_8steps_wansyn_${RUN_ID}"}

training_args=(
  --tracker_project_name "phase1_wan_dmd2_8steps_wansyn"
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 448
  --num_width 832
  --num_frames 77
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
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 2e-6
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
  --dmd_denoising_steps '1000,850,700,550,350,275,200,125'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3.5
)

torchrun \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  fastvideo/training/distillation.py \
  --distill_model "wan" \
  --distill_method "dmd2" \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${miscellaneous_args[@]}" \
  "${dmd_args[@]}"
