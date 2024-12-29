export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

CUDA_VISIBLE_DEVICES=6 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29403 \
    fastvideo/train_new.py \
    --seed 42 \
    --model_type mochi \
    --pretrained_model_name_or_path ~/data/mochi_diffusers \
    --cache_dir data/.cache \
    --data_json_path data/Encoder_Overfit_Data/videos2caption.json \
    --validation_prompt_dir data/validation_prompt_embed_mask \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 2 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 2000 \
    --learning_rate 5e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 200 \
    --validation_steps 100 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir data/outputs/Black-Myth-Lora-FT \
    --tracker_project_name Black-Myth-Lora-Finetune \
    --num_frames 91 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --master_weight_type fp32 \
    --use_lora \
    --use_cpu_offload
