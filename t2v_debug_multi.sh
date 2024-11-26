export WANDB_MODE=online
export WANDB_API_KEY="4f6de3765d6464f43e0506ec7d785641af645e73"

torchrun --nnodes 1 --nproc_per_node 4\
    fastvideo/distill.py\
    --seed 42\
    --pretrained_model_name_or_path data/mochi\
    --dit_model_name_or_path data/Mochi-Image\
    --cache_dir "data/.cache"\
    --data_json_path "data/Image-Train-Dataset/videos2caption.json"\
    --validation_prompt_dir "data/Encoder_Overfit_Data/validation_prompt_embed_mask"\
    --uncond_prompt_dir "data/BLACK-MYTH-YQ/uncond_prompt_embed_mask"\
    --gradient_checkpointing\
    --train_batch_size=8\
    --num_latent_t 1\
    --sp_size 1\
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps=20000\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=250\
    --validation_steps 50\
    --validation_sampling_steps 4 \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.1\
    --ema_decay 0.999\
    --log_validation\
    --output_dir="data/outputs/image_distill_4.5_no_cfg_distill"\
    --tracker_project_name PCM \
    --num_frames 1 \
    --shift 8.0 \
    --validation_guidance_scale 4.5 \
    --not_apply_cfg_solver