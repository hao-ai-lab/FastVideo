#!/bin/bash
#SBATCH --job-name=v-t-2
#SBATCH --partition=main
#SBATCH --qos=hao
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=vsa-t2v/2e5.out
#SBATCH --error=vsa-t2v/2e5.err
#SBATCH --exclusive
set -e -x

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn_77x768x1280/latents_i2v/train/
VALIDATION_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn_77x768x1280/latents_i2v/test_64/
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

OUTPUT_PATH="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn_77x768x1280/VSA_T2V_14B_2e5_bs32"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29501
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
srun torchrun --nnodes 8 --nproc_per_node $NUM_GPUS \
    --node_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/v1/training/wan_training_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_preprocessed_path "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 20 \
    --sp_size 2 \
    --tp_size 2 \
    --num_gpus 64 \
    --hsdp_replicate_dim 8 \
    --hsdp-shard-dim 8  \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 3000 \
    --learning_rate 2e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 600 \
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
    --num_height 768 \
    --num_width 1280 \
    --num_frames  77 \
    --flow_shift 1 \
    --validation_guidance_scale "5.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --VSA_decay_rate 0.03 \
    --VSA_decay_interval_steps 30 \
    --VSA_sparsity 0.9 \
    --gradient_checkpointing True \
    --gradient_checkpointing_type "full"
