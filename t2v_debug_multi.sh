export WANDB_MODE=online
torchrun --nnodes 1 --nproc_per_node 4 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/Encoder_Overfit_Data/videos2caption.json" \
    --validation_prompt_dir "data/Encoder_Overfit_Data/validation_prompt_embed_mask" \
    --uncond_prompt_dir "data/Encoder_Overfit_Dataa/uncond_prompt_embed_mask" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 14 \
    --sp_size 2 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=2000 \
    --learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=250 \
    --validation_steps 250 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir="data/outputs/T_J_FT" \
    --weighting_scheme "uniform" 

