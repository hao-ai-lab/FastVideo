#!/bin/bash
#SBATCH --job-name=t2v
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_t2v_output/t2v_%j.out
#SBATCH --error=dmd_t2v_output/t2v_%j.err
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29503
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=your_wandb_api_key
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

# Configs
NUM_GPUS=1

# Model paths for Self-Forcing DMD distillation:
GENERATOR_MODEL_PATH="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"  # Teacher model
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Critic model

DATA_DIR=your_data_dir
VALIDATION_DATASET_FILE=your_validation_data_dir
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

torchrun \
--nnodes 1 \
--master_port $MASTER_PORT \
--nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_self_forcing_distillation_pipeline.py \
    --config examples/distill/SFWan2.1-T2V/distill_dmd_t2v_1.3B.yaml \
    --model_path "$GENERATOR_MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.distillation.real_score_model_path "$REAL_SCORE_MODEL_PATH" \
    --training.distillation.fake_score_model_path "$FAKE_SCORE_MODEL_PATH" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
