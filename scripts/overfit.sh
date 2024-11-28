accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_merge_path "data/Mochi-Synthetic-Data/merge.txt" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 14 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=200 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --checkpointing_steps=50 \
    --allow_tf32 \
    --tile_sample_stride 192 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --enable_tiling \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --output_dir="data/outputs/debug_1e-5"

python fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface. The plastic wrap is stretched to cover the dough more securely. The hand adjusts the wrap, ensuring that it is tight and smooth over the dough. The scene focuses on the hand's movements as it secures the edges of the plastic wrap. No new objects appear, and the camera remains stationary, focusing on the action of covering the dough." \
    --num_frames 79 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --seed 12346 \
    --transformer_path data/outputs/debug_1e-5/checkpoint-50/transformer \
    --output_path 123.mp4