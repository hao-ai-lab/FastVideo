#!/bin/bash
#SBATCH --job-name=i2v
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=i2v_output/i2v_%j.out
#SBATCH --error=i2v_output/i2v_%j.err
#SBATCH --exclusive
set -e -x

# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate will-fv

# Basic Info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=8

MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
DATA_DIR="data/crush-smol_processed_i2v_1_3b_inp/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/validation.json"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t

# Training arguments
training_args=(
  --tracker_project_name wan_i2v_finetune
  --output_dir "checkpoints/wan_i2v_finetune"
  --max_train_steps 2000
  --train_batch_size 2
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
  --sp_size $NUM_GPUS
  --tp_size $NUM_GPUS
  --hsdp_replicate_dim $SLURM_JOB_NUM_NODES
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
  --dataloader_num_workers 10
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "40"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
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
  --enable_gradient_checkpointing_type "full"
)

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_i2v_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"