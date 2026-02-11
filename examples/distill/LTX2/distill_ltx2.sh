#!/bin/bash
#SBATCH --job-name=ltx2_distillation
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/ltx2_distillation_%j.out
#SBATCH --error=logs/ltx2_distillation_%j.err

source .venv/bin/activate

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

# Use node-local Triton cache to avoid stale file handle errors on shared filesystems
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_NODEID}"
export WANDB_API_KEY=50632ebd88ffd970521cec9ab4a1a2d7e85bfc45
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

set -euo pipefail

# ---- torchrun rendezvous (multi-node) ----
# 1. Get the hostname of the first node (Master)
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
# MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
MASTER_ADDR=${nodes[0]}
MASTER_PORT=29500

# 2. Get the node count automatically
NNODES=8
GPUS_PER_NODE=4
NUM_GPUS=32

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES"

# Configs
MODEL_PATH="Davids048/LTX2-Base-Diffusers"
REAL_SCORE_MODEL_PATH="Davids048/LTX2-Base-Diffusers"
FAKE_SCORE_MODEL_PATH="Davids048/LTX2-Base-Diffusers"
DATA_DIR=/home/hal-shared/ltx2-data/
VALIDATION_DIR="examples/distill/LTX2/validation.json"
OUTPUT_DIR="ltx2_distill_8steps"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name LTX2_distillation
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 31
  --num_height 1088
  --num_width 1920
  --num_frames 121
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
  --validation_dataset_file "$VALIDATION_DIR"
  --validation_steps 50
  --validation_sampling_steps "8"
  --validation_guidance_scale "1.0" # used by validation inference; keep aligned with basic_ltx2_distilled defaults
  --text-encoder-cpu-offload
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --use_ema True
  --ema_decay 0.99
  --ema_start_step 200
  --flow_shift 5  # TODO: need to determine the correct value
  --seed 1000
)

# DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,993,987,981,975,909,725,421'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3
  --fake_score_learning_rate 2e-6
  --fake_score_betas '0.0,0.999'
  --simulate_generator_forward 
  --log_visualization # disable if oom
)

srun torchrun \
    --nnodes $NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      fastvideo/training/ltx2_distillation_pipeline.py \
      "${parallel_args[@]}" \
      "${model_args[@]}" \
      "${dataset_args[@]}" \
      "${training_args[@]}" \
      "${optimizer_args[@]}" \
      "${validation_args[@]}" \
      "${miscellaneous_args[@]}" \
      "${dmd_args[@]}"
