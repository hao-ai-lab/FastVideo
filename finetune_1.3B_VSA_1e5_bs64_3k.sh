#!/bin/bash
#SBATCH --job-name=v-i-1
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=vsa-i2v/1.3B-1e5.out
#SBATCH --error=vsa-i2v/1.3B-1e5.err
#SBATCH --exclusive
set -e -x

# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate will-fv

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# will key
export WANDB_API_KEY='8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
# DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents_i2v/train/
DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents_i2v/test_filter/
VALIDATION_DIR=/mnt/weka/home/hao.zhang/wl/FastVideo/data/mixkit/validation.json
NUM_GPUS=8
# export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# OUTPUT_PATH="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn_77x768x1280/VSA_I2V_1.3B_1e5_bs64"
OUTPUT_PATH="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn_77x768x1280/VSA_I2V_1.3B_1e5_bs32"
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
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
srun torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    --node_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/v1/training/wan_i2v_training_pipeline.py \
    --model_path $MODEL_PATH \
    --inference_mode False\
    --pretrained_model_name_or_path $MODEL_PATH \
    --data_path "$DATA_DIR" \
    --validation_dataset_file "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 16 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus 8 \
    --hsdp_replicate_dim 8 \
    --hsdp-shard-dim 1  \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 4500 \
    --learning_rate 2e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 1000 \
    --validation_steps 300 \
    --validation_sampling_steps "50" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --training_cfg_rate 0.1 \
    --seed 1024 \
    --output_dir $OUTPUT_PATH \
    --tracker_project_name VSA_finetune \
    --num_height 448 \
    --num_width 832 \
    --num_frames  61 \
    --flow_shift 3 \
    --validation_guidance_scale "6.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --VSA_decay_rate 0.03 \
    --VSA_decay_interval_steps 50 \
    --VSA_sparsity 0.9 
