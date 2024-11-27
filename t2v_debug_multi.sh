export WANDB_MODE=online

torchrun --nnodes 4 --nproc_per_node 4\
    --node_rank=3 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=172.23.30.15:29500 \
    fastvideo/distill.py\
    --seed 42\
    --pretrained_model_name_or_path data/mochi\
    --cache_dir "data/.cache"\
    --data_json_path "data/General-Video/videos2caption.json"\
    --validation_prompt_dir "data/Encoder_Overfit_Data/validation_prompt_embed_mask"\
    --uncond_prompt_dir "data/Encoder_Overfit_Data/uncond_prompt_embed_mask"\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 28\
    --sp_size 4\
    --train_sp_batch_size 2\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps=20000\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=250\
    --validation_steps 25\
    --validation_sampling_steps 8 \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --ema_decay 0.999\
    --log_validation\
    --output_dir="data/outputs/video_distill_shift_8"\
    --tracker_project_name PCM \
    --num_frames 163 \
    --shift 8.0 \
    --validation_guidance_scale 4.5 
    

