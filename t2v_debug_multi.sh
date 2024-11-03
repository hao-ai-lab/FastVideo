accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_merge_path "data/synthetic_debug/merge.txt" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 14 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=10 \
    --mixed_precision="bf16" \
    --checkpointing_steps=50 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --output_dir="data/outputs/debug"




accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_merge_path "data/synthetic_debug/merge.txt" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 28 \
    --sp_size 4 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100 \
    --learning_rate=2e-5 \
    --lr_warmup_steps=10 \
    --mixed_precision="bf16" \
    --checkpointing_steps=50 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --output_dir="data/outputs/debug_sp"



torchrun --nnodes=1 --nproc_per_node=4 --master_port 29503 debug_mochi_sp.py
torchrun --nnodes=1 --nproc_per_node=4 --master_port 29503  debug_OSP_A2A.py