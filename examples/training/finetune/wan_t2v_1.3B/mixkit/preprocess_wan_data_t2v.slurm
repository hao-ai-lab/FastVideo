#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=preprocess_mixkit_wan_1.3b_t2v/preprocess_wan_data_t2v.log
#SBATCH --error=preprocess_mixkit_wan_1.3b_t2v/preprocess_wan_data_t2v.error
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG_SUBSYS=INIT,NET
# different cache dir for different processes
# export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false
# export WANDB_API_KEY="8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde"
export WANDB_API_KEY="50632ebd88ffd970521cec9ab4a1a2d7e85bfc45"
# export WANDB_API_KEY='your_wandb_api_key_here'
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

export HF_HOME=/mnt/data/huggingface
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/shared-bin/local/usr/lib/aarch64-linux-gnu:$CUDA_HOME/lib64"
export CUDACXX="$CUDA_HOME/bin/nvcc"

source ~/miniconda3/bin/activate
conda activate fastvideo

GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="/mnt/data/mixkit/merge.txt"
OUTPUT_DIR="/mnt/data/mixkit_wan_1.3b_processed_t2v/"

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $GPU_NUM \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 1 \
    --seed 42 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 81 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR \
    --train_fps 16 \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --drop_short_ratio 0.0 \
    --preprocess_task "t2v" 