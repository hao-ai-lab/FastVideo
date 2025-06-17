export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=data/mini_dataset_i2v_VSA/combined_parquet_dataset
VALIDATION_DIR=data/mini_dataset_i2v_VSA/validation_parquet_dataset
NUM_GPUS=1
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
    --num_latent_t 4 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim $NUM_GPUS \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 5 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 6000 \
    --validation_steps 10 \
    --validation_sampling_steps "50" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --output_dir "$DATA_DIR/outputs/wan_finetune" \
    --tracker_project_name VSA_finetune \
    --num_height 384 \
    --num_width 512 \
    --num_frames  13 \
    --flow_shift 3 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --VSA_decay_rate 0.01 \
    --VSA_decay_interval_steps 1 \
    --VSA_sparsity 0.9
# --resume_from_checkpoint "$CHECKPOINT_PATH"
