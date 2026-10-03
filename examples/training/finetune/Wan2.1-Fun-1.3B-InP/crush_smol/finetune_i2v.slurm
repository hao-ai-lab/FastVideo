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

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_i2v_training_pipeline.py \
    --config examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/finetune_i2v_slurm.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.sp_size "$NUM_GPUS" \
    --engine.parallelism.tp_size "$NUM_GPUS" \
    --engine.parallelism.hsdp_replicate_dim "$SLURM_JOB_NUM_NODES" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"