export WANDB_MODE=online
export WANDB_API_KEY=[WANDB_API_KEY]
# This can avoid OOM
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

torchrun --nnodes 2 --nproc_per_node 8\
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=[MASTER_NODE_IP_ADDRESS]:29500 \
    fastvideo/distill_adv.py\
    --seed 42\
    --pretrained_model_name_or_path data/mochi\
    --cache_dir "data/.cache"\
    --data_json_path "data/Merge-30k-Data/video2caption.json"\
    --validation_prompt_dir "data/validation_embeddings/validation_prompt_embed_mask"\
    --uncond_prompt_dir "data/validation_embeddings/uncond_prompt_embed_mask"\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 28\
    --sp_size 4\
    --train_sp_batch_size 2\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps=4000\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=250\
    --validation_steps 125\
    --validation_sampling_steps 8 \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --ema_decay 0.999\
    --log_validation\
    --output_dir="data/outputs/adv_shift8_euler_100"\
    --tracker_project_name PCM \
    --num_frames  163 \
    --shift 8.0 \
    --validation_guidance_scale 4.5 


