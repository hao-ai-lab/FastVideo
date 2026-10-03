#!/bin/bash
#SBATCH --job-name=i2v
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=VSA_i2v_output/i2v_%j.out
#SBATCH --error=VSA_i2v_output/i2v_%j.err
#SBATCH --exclusive
set -e -x

# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate your_env

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
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=8
MODEL_PATH="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
DATA_DIR=your_data_dir
VALIDATION_DATASET_FILE=your_validation_dataset_file
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_training_pipeline.py \
    --config examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/I2V-14B-VSA.yaml \
    --model_path "$MODEL_PATH" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
