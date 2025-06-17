export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=PY007/crush-smol/combined_parquet_dataset
VALIDATION_DIR=PY007/crush-smol/validation_parquet_dataset
NUM_GPUS=4
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
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
    --train_batch_size 2 \
    --num_latent_t 4 \
    --sp_size $NUM_GPUS \
    --tp_size $NUM_GPUS \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim 1 \
    --hsdp-shard-dim $NUM_GPUS \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 5 \
    --learning_rate 1e-6 \
    --mixed_precision "bf16" \
    --checkpointing_steps 30 \
    --validation_steps 10 \
    --validation_sampling_steps "8" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --output_dir "data/wan_finetune_test" \
    --tracker_project_name wan_finetune_ci \
    --num_height 480 \
    --num_width 832 \
    --num_frames  81 \
    --flow_shift 3 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --not_apply_cfg_solver \
# --resume_from_checkpoint "$CHECKPOINT_PATH"
