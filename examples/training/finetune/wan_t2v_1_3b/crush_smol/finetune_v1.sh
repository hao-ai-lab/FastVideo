export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
VALIDATION_DIR="data/crush-smol_processed_t2v/validation_parquet_dataset/"
NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS\
    fastvideo/v1/training/wan_training_pipeline.py\
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --data_path "$DATA_DIR"\
    --validation_preprocessed_path "$VALIDATION_DIR"\
    --train_batch_size=1 \
    --num_latent_t 8 \
    --sp_size 8 \
    --tp_size 8 \
    --hsdp_replicate_dim 1 \
    --hsdp_shard_dim 8 \
    --num_gpus $NUM_GPUS \
    --train_sp_batch_size 1\
    --dataloader_num_workers 10\
    --gradient_accumulation_steps=8 \
    --max_train_steps=5000 \
    --learning_rate=1e-5\
    --mixed_precision="bf16"\
    --checkpointing_steps=6000 \
    --validation_steps 100\
    --validation_sampling_steps "50" \
    --log_validation \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --output_dir="$DATA_DIR/outputs/wan_t2v_finetune"\
    --tracker_project_name wan_t2v_finetune \
    --num_height 480 \
    --num_width 832 \
    --num_frames  77 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --weight_decay 0.01 \
    --not_apply_cfg_solver \
    --dit_precision "fp32" \
    --max_grad_norm 1.0
