export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

DATA_DIR=./data

torchrun --nnodes 1 --nproc_per_node 8\
    fastvideo/distill.py\
    --seed 42\
    --pretrained_model_name_or_path $DATA_DIR/hunyuan\
    --dit_model_name_or_path $DATA_DIR/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan" \
    --cache_dir "$DATA_DIR/.cache"\
    --data_json_path "$DATA_DIR/Hunyuan-30K-Distill-Data/videos2caption.json"\
    --validation_prompt_dir "$DATA_DIR/Hunyuan-Distill-Data/validation"\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 24\
    --sp_size 1\
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps=2000\
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
    --output_dir="$DATA_DIR/outputs/hy_phase1_shift17_bs_32"\
    --tracker_project_name Hunyuan_Distill \
    --num_frames  93 \
    --shift 17 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver 