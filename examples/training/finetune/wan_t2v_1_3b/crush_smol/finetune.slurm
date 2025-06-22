#!/bin/bash
#SBATCH --job-name=FV_2N_14B
#SBATCH --partition=main
#SBATCH --qos=hao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --nodelist=fs-mbz-gpu-[400-550]
#SBATCH --mem=1440G
#SBATCH --output=4n_i2v/4n_i2v_%j.out
#SBATCH --error=4n_i2v/4n_i2v_%j.err
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
export WANDB_API_KEY='8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde'

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=8

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

DATA_DIR=data/crush-smol_processed/combined_parquet_dataset
VALIDATION_DIR=data/crush-smol_processed/validation_parquet_dataset
NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/v1/training/wan_i2v_training_pipeline.py\
    --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --cache_dir "/home/ray/.cache"\
    --data_path "$DATA_DIR"\
    --validation_preprocessed_path "$VALIDATION_DIR"\
    --train_batch_size=1\
    --num_latent_t 8 \
    --num_gpus $NUM_GPUS \
    --sp_size $NUM_GPUS \
    --tp_size $NUM_GPUS \
    --hsdp_replicate_dim $SLURM_JOB_NUM_NODES \
    --hsdp_shard_dim $NUM_GPUS \
    --train_sp_batch_size 1\
    --dataloader_num_workers 10\
    --gradient_accumulation_steps=1\
    --max_train_steps=10000 \
    --learning_rate=5e-5\
    --mixed_precision="bf16"\
    --checkpointing_steps=11000 \
    --validation_steps 100\
    --validation_sampling_steps "40" \
    --log_validation \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --output_dir="$DATA_DIR/outputs/wan_i2v_finetune_2n"\
    --tracker_project_name wan_i2v_finetune \
    --num_height 480 \
    --num_width 832 \
    --num_frames 77 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --weight_decay 1e-4 \
    --not_apply_cfg_solver \
    --dit_precision "fp32" \
    --max_grad_norm 1.0