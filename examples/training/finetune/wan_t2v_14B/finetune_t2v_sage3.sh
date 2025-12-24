#!/bin/bash
#SBATCH --job-name=wan_t2v_14B_finetune_sage3
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/wan_t2v_14B_finetune_sage3_%j.out
#SBATCH --error=logs/wan_t2v_14B_finetune_sage3_%j.err

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# export TRITON_PRINT_AUTOTUNING=1  # to print the best config
export WANDB_API_KEY=2f25ad37933894dbf0966c838c0b8494987f9f2f
MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
DATA_DIR=data/Wan-Syn_77x448x832_600k
VALIDATION_DATASET_FILE="examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json"
NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5

set -euo pipefail

# ---- torchrun rendezvous (multi-node) ----
# Launch ONE torchrun per node (via srun) and let torchrun spawn 4 workers per node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT


# Training arguments
training_args=(
  --tracker_project_name "wan_t2v_finetune_qat"
  --output_dir "checkpoints/wan_t2v_finetune_sage3_14B"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 720
  --num_width 1280
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --generator_4bit_attn True
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS 
  --sp_size 8
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
  --data_path $DATA_DIR
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation 
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 200
  --validation_sampling_steps "50" 
  --validation_guidance_scale "5.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 5
  --seed 1000
)

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  torchrun \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node 4 \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --node_rank "$SLURM_NODEID" \
      fastvideo/training/wan_training_pipeline.py \
      "${parallel_args[@]}" \
      "${model_args[@]}" \
      "${dataset_args[@]}" \
      "${training_args[@]}" \
      "${optimizer_args[@]}" \
      "${validation_args[@]}" \
      "${miscellaneous_args[@]}"
