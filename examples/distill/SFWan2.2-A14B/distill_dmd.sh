#!/bin/bash
#SBATCH --job-name=t2v
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_t2v_output/t2v_%j.out
#SBATCH --error=dmd_t2v_output/t2v_%j.err
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG_SUBSYS=INIT,NET
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_API_KEY='your_wandb_api_key_here'
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

# Configs
NUM_GPUS=8

# Model paths for Self-Forcing DMD distillation with Wan2.2:
# GENERATOR_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"  # Updated to Wan2.2
# REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers" # Teacher model
# FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers" # Critic model
GENERATOR_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Updated to Wan2.2
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Teacher model
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" # Critic model

# DATA_DIR="data/test-text-preprocessing/Node_0_GPU_1_File_1/combined_parquet_dataset/"
DATA_DIR="/mnt/weka/home/hao.zhang/matthew/FastVideo/data/test-text-preprocessing"
# DATA_DIR=data/crush-smol_processed_t2v/combined_parquet_dataset
# DATA_DIR="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn-upload/latents_i2v/train/"
# VALIDATION_DATASET_FILE="data/crush-smol-single_processed_t2v/validation.json"
VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wl/FastVideo/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/validation_64.json"

# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_self_forcing_distillation_pipeline.py \
    --config examples/distill/SFWan2.2-A14B/distill_dmd.yaml \
    --model_path "$GENERATOR_MODEL_PATH" \
    --training.distillation.real_score_model_path "$REAL_SCORE_MODEL_PATH" \
    --training.distillation.fake_score_model_path "$FAKE_SCORE_MODEL_PATH" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"