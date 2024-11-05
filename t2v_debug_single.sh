# export WANDB_MODE="offline"

accelerate launch \
    --config_file scripts/accelerate_configs/single.yaml \
    fastvideo/train.py \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_merge_path "data/Mochi-Synthetic-Data/merge.txt" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 14 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --optimizer="AdamW" \
    --mixed_precision="bf16" \
    --checkpointing_steps=250 \
    --allow_tf32 \
    --tile_sample_stride 192 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --resume_from_checkpoint="latest" \
    --ema_decay 0.999 \
    --enable_tiling \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --output_dir="data/outputs/debug" \