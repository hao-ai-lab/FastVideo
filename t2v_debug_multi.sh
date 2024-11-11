torchrun --nnodes 1 --nproc_per_node 4 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/Encoder_Overfit_Data/videos2caption.json" \
    --validation_prompt_dir "data/Encoder_Overfit_Data/validation_prompt_embed_mask" \
    --uncond_prompt_dir "data/Encoder_Overfit_Data/uncond_prompt_embed_mask" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 2 \
    --sp_size 2 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 10 \
    --learning_rate 1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps 2 \
    --validation_steps 2 \
    --validation_sampling_steps 2 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir="data/outputs/BW_Testrun" \
    --use_lora \

torchrun --nnodes 1 --nproc_per_node 4 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir data/.cache \
    --data_json_path data/Encoder_Overfit_Data/videos2caption.json \
    --validation_prompt_dir data/Encoder_Overfit_Data/validation_prompt_embed_mask \
    --uncond_prompt_dir data/Encoder_Overfit_Data/uncond_prompt_embed_mask \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 14 \
    --sp_size 2 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 500 \
    --learning_rate 1e-4 \
    --mixed_precision bf16 \
    --checkpointing_steps 100 \
    --validation_steps 50 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir data/outputs/BW_Testrun_2/ \
    --lora_rank 128 \
    --lora_alpha 256 \
    --use_lora




# SP single node: 13
# SP multi node: 20
# SP multi node with larger batch size: 15
# No SP Single node: 10.5
# No SP Multe node: 11



