export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

torchrun --nnodes 1 --nproc_per_node 4 \
    fastvideo/distill.py \
    --seed 42 \
    --pretrained_model_name_or_path data/FastMochi \
    --model_type "mochi" \
    --cache_dir data/.cache \
    --data_json_path data/Merge-30k-Data/video2caption.json \
    --validation_prompt_dir data/Image-Vid-Finetune-Mochi/validation \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 28 \
    --sp_size 4 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=4000 \
    --learning_rate=1e-6 \
    --mixed_precision=bf16 \
    --checkpointing_steps=64 \
    --validation_steps 1 \
    --validation_sampling_steps 8 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir="data/outputs/lq_euler_50_thres0.1_lrg_0.75_phase1_lr1e-6_repro" \
    --tracker_project_name PCM \
    --num_frames 163 \
    --scheduler_type pcm_linear_quadratic \
    --validation_guidance_scale 0.5,1.5,2.5 \
    --num_euler_timesteps 50 \
    --linear_quadratic_threshold 0.1 \
    --linear_range 0.75 \
    --multi_phased_distill_schedule 4000-1