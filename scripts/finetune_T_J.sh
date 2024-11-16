torchrun --nnodes 4 --nproc_per_node 4 \
    --node_rank=3 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=172.23.30.15:29500 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/BLACK-MYTH-Finetune-Dataset/videos2caption.json" \
    --validation_prompt_dir "data/BLACK-MYTH-Finetune-Dataset/validation_prompt_embed_mask" \
    --uncond_prompt_dir "data/BLACK-MYTH-Finetune-Dataset/uncond_prompt_embed_mask" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 16 \
    --sp_size 4 \
    --train_sp_batch_size 4 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=4000 \
    --learning_rate=2e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=500 \
    --validation_steps=100 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir="data/outputs/black_myth_correct_video_mask" \
    --weighting_scheme "uniform" \
    --num_frames 91 \
    --selective_checkpointing 1.0


torchrun --nnodes 4 --nproc_per_node 4 \
    --node_rank=3 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=172.23.30.15:29500 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/BLACK-MYTH-DREAM-BOOTH/videos2caption.json" \
    --validation_prompt_dir "data/BLACK-MYTH-DREAM-BOOTH/validation_prompt_embed_mask" \
    --uncond_prompt_dir "data/BLACK-MYTH-DREAM-BOOTH/uncond_prompt_embed_mask" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 16 \
    --sp_size 4 \
    --train_sp_batch_size 4 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=4000 \
    --learning_rate=2e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=100 \
    --validation_steps=50 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir="data/outputs/black_myth_correct_video_mask" \
    --weighting_scheme "uniform" \
    --num_frames 91 \
    --selective_checkpointing 1.0




torchrun --nnodes 1 --nproc_per_node 4 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/BLACK-MYTH-Finetune-Dataset/videos2caption.json" \
    --validation_prompt_dir "data/BLACK-MYTH-Finetune-Dataset/validation_prompt_embed_mask" \
    --uncond_prompt_dir "data/BLACK-MYTH-Finetune-Dataset/uncond_prompt_embed_mask" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 16 \
    --sp_size 4 \
    --train_sp_batch_size 4 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2000 \
    --learning_rate=2e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=100 \
    --validation_steps=100 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir="data/outputs/black_myth_dream" \
    --weighting_scheme "uniform" \
    --num_frames 91 \


| 4/2000 [01:23<10:59:33, 19.83s/it, loss=0.74, grad_norm=2.6]


| 2/2000 [00:44<11:59:27, 21.61s/it, loss=0.873, grad_norm=1.57]
| 3/2000 [01:03<11:18:51, 20.40s/it, loss=0.813, grad_norm=1.48]


# FSDP Hybrid: 

