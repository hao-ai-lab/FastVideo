export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

DATA_DIR=./data
# IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
    # --gradient_checkpointing\
    # --pretrained_model_name_or_path hunyuanvideo-community/HunyuanVideo \
    # --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
torchrun --nnodes 1 --nproc_per_node 4\
    fastvideo/v1/pipelines/training_pipeline.py\
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/ray/.cache"\
    --data_json_path "$DATA_DIR/HD-Mixkit-Finetune-Hunyuan/videos2caption.json"\
    --validation_prompt_dir "$DATA_DIR/HD-Mixkit-Finetune-Hunyuan/validation"\
    --train_batch_size=1\
    --num_latent_t 1 \
    --sp_size 4 \
    --tp_size 4 \
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps=320\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=64\
    --validation_steps 64\
    --validation_sampling_steps "2,4,8" \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --log_validation\
    --output_dir="$DATA_DIR/outputs/hy_phase1_shift17_bs_16_HD"\
    --tracker_project_name Hunyuan_Distill \
    --num_height 720 \
    --num_width 1280 \
    --num_frames  125 \
    --shift 17 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver \
    --master_weight_type "bf16"