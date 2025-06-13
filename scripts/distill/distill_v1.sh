export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

DATA_DIR=data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset
VALIDATION_DIR=data/HD-Mixkit-Finetune-Wan/validation_parquet_dataset
num_gpus=1
# IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
    # --gradient_checkpointing\
    # --pretrained_model_name_or_path hunyuanvideo-community/HunyuanVideo \
    # --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
torchrun --nnodes 1 --nproc_per_node $num_gpus\
    fastvideo/v1/training/wan_distillation_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --mode distill \
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/test/.cache" \
    --data_path "$DATA_DIR" \
    --validation_prompt_dir "$VALIDATION_DIR" \
    --train_batch_size=1 \
    --num_latent_t 4 \
    --sp_size $num_gpus \
    --dp_size $num_gpus \
    --dp_shards $num_gpus \
    --train_sp_batch_size 1 \
    --dataloader_num_workers $num_gpus \
    --gradient_accumulation_steps=1 \
    --max_train_steps=540 \
    --learning_rate=1e-6 \
    --mixed_precision="bf16" \
    --checkpointing_steps=64 \
    --validation_steps 180 \
    --validation_sampling_steps "2,4,8" \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir="$DATA_DIR/outputs/hy_phase1_shift17_bs_16_HD" \
    --tracker_project_name Hunyuan_Distill \
    --num_height 720 \
    --num_width 1280 \
    --num_frames  81 \
    --shift 17 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver \
    --weight_decay 0.01 \
    --master_weight_type "fp32" \
    --distill_cfg 3.0 \
    --pred_decay_weight 0.0 \
    --max_grad_norm 1.0
    # --master_weight_type "bf16"