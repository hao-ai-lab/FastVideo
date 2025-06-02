export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

DATA_DIR=data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset
VALIDATION_DIR=data/HD-Mixkit-Finetune-Wan/validation_parquet_dataset
NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

CHECKPOINT_PATH="$DATA_DIR/outputs/wan_finetune/checkpoint-5"

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS\
    fastvideo/v1/training/wan_training_pipeline.py\
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/ray/.cache"\
    --data_path "$DATA_DIR"\
    --validation_prompt_dir "$VALIDATION_DIR"\
    --train_batch_size=1\
    --num_latent_t 4 \
    --sp_size $NUM_GPUS \
    --tp_size $NUM_GPUS \
    --train_sp_batch_size 1\
    --dataloader_num_workers 5\
    --gradient_accumulation_steps=1\
    --max_train_steps=120 \
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=50 \
    --validation_steps 20\
    --validation_sampling_steps "2,4,8" \
    --log_validation \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --output_dir="$DATA_DIR/outputs/wan_finetune"\
    --tracker_project_name wan_finetune \
    --num_height 480 \
    --num_width 832 \
    --num_frames  81 \
    --shift 3 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --weight_decay 0.01 \
    --not_apply_cfg_solver \
    --master_weight_type "fp32" \
    --max_grad_norm 1.0 \

# --resume_from_checkpoint "$CHECKPOINT_PATH"
