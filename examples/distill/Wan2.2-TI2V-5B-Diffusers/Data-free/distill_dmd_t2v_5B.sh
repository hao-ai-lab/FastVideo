#!/bin/bash
#SBATCH --job-name=t2v
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_Wan2.2/t2v_g2e5_f1e5_%j.out
#SBATCH --error=dmd_Wan2.2/t2v_g2e5_f1e5_%j.err
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
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_API_KEY=your_wandb_api_key
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=8
MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DATA_DIR=your_data_dir
VALIDATION_DIR=your_validation_path  #(example:validation_64.json)
OUTPUT_DIR="checkpoints/wan_t2v_finetune"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_distillation_pipeline.py \
    --config examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/distill_dmd_t2v_5B.yaml \
    --model_path "$MODEL_PATH" \
    --training.distillation.real_score_model_path "$REAL_SCORE_MODEL_PATH" \
    --training.distillation.fake_score_model_path "$FAKE_SCORE_MODEL_PATH" \
    --training.data.data_path "$DATA_DIR" \
    --training.checkpoint.output_dir "$OUTPUT_DIR" \
    --training.validation.dataset_file "$VALIDATION_DIR"
