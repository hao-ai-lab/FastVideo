#!/bin/bash
#SBATCH --job-name=VSA-1.3B-1n
#SBATCH --partition=main
#SBATCH --qos=hao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=960G
#SBATCH --output=VSA-1.3B-1n/sparse_ft_VSA-1.3B-1n_%j.out
#SBATCH --error=VSA-1.3B-1n/sparse_ft_VSA-1.3B-1n_%j.err
#SBATCH --exclusive
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents/train_10K/
VALIDATION_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents/test/
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

CHECKPOINT_PATH="$DATA_DIR/outputs/wan_finetune/checkpoint-5"

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/v1/training/wan_training_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_prompt_dir "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 20 \
    --sp_size 2 \
    --tp_size 2 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim $NUM_GPUS \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 30000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 6000 \
    --validation_steps 100 \
    --validation_sampling_steps "50" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --output_dir "$DATA_DIR/outputs/wan_finetune" \
    --tracker_project_name VSA_finetune \
    --num_height 448 \
    --num_width 832 \
    --num_frames  77 \
    --shift 3 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --VSA_decay_sparsity 0.9 \
    --VSA_decay_rate 0.02 \
    --VSA_decay_interval_steps 50 \
    --VSA_val_sparsity 0.9