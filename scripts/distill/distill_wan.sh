#https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/main
DATA_DIR=./data

torchrun --nnodes 1 --nproc_per_node 8\
    fastvideo/distill.py\
    --seed 42\
    --pretrained_model_name_or_path $DATA_DIR/wan\
    --model_type "wan" \
    --cache_dir "$DATA_DIR/.cache"\
    --data_json_path "$DATA_DIR/Image-Vid-Finetune-Wan/videos2caption.json"\
    --validation_prompt_dir "$DATA_DIR/Image-Vid-Finetune-Wan/validation"\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 32 \
    --sp_size 1 \
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps=320\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=64\
    --validation_steps 64\
    --validation_sampling_steps "8" \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 5.0\
    --log_validation\
    --output_dir="$DATA_DIR/outputs/wan"\
    --tracker_project_name Wan_PCM \
    --num_height 480 \
    --num_width 832 \
    --num_frames  81 \
    --shift 3 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --not_apply_cfg_solver 