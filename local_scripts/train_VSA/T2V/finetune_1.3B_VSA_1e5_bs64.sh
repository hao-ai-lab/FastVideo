#!/bin/bash
#SBATCH --job-name=v-t-3
#SBATCH --partition=main
#SBATCH --qos=hao
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=vsa-t2v/1.3B-1e5.out
#SBATCH --error=vsa-t2v/1.3B-1e5.err
#SBATCH --exclusive
set -e -x

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents_i2v/train/
VALIDATION_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents_i2v/test_32/
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

OUTPUT_PATH="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/VSA_T2V_1.3B_1e5_bs64_0.8_3k"
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

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
srun torchrun --nnodes 4 --nproc_per_node $NUM_GPUS \
    --node_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/v1/training/wan_training_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_preprocessed_path "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 16 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus 32 \
    --hsdp_replicate_dim 32 \
    --hsdp-shard-dim 1  \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 3000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 200 \
    --validation_steps 200 \
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
    --flow_shift 1 \
    --validation_guidance_scale "6.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --VSA_decay_rate 0.04 \
    --VSA_decay_interval_steps 50 \
    --VSA_sparsity 0.8 \
