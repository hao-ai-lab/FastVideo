#!/bin/bash
#SBATCH --job-name=FV_2N_14B
#SBATCH --partition=main
#SBATCH --qos=hao
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --nodelist=fs-mbz-gpu-[400-550]
#SBATCH --mem=1440G
#SBATCH --output=fv_2n_test/fv_2n_test_%j.out
#SBATCH --error=fv_2n_test/fv_2n_test_%j.err
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

DATA_DIR=data/cats_2/combined_parquet_dataset
VALIDATION_DIR=data/cats_2/validation_parquet_dataset
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
    fastvideo/v1/training/wan_training_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --cache_dir "/home/ray/.cache"\
    --data_path "$DATA_DIR"\
    --validation_prompt_dir "$VALIDATION_DIR"\
    --train_batch_size=1\
    --num_latent_t 16 \
    --sp_size $NUM_GPUS \
    --tp_size $NUM_GPUS \
    --dp_size $SLURM_JOB_NUM_NODES \
    --dp_shards $NUM_GPUS \
    --train_sp_batch_size 1\
    --dataloader_num_workers 5\
    --gradient_accumulation_steps=1\
    --max_train_steps=500 \
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=2000 \
    --validation_steps 10\
    --validation_sampling_steps "2,4,8" \
    --log_validation \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --output_dir="$DATA_DIR/outputs/wan_finetune"\
    --tracker_project_name wan_finetune_m2 \
    --num_height 480 \
    --num_width 832 \
    --num_frames  81 \
    --shift 3 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --weight_decay 0.01 \
    --not_apply_cfg_solver \
    --master_weight_type "fp32" \
    --max_grad_norm 1.0
