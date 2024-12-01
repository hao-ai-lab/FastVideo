export WANDB_MODE=online
export WANDB_API_KEY=4f6de3765d6464f43e0506ec7d785641af645e73
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_PROTO=simple

torchrun --nnodes 2 --nproc_per_node 8\
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=[MASTER_NODE_IP_ADDRESS]:29500 \
    fastvideo/distill.py\
    --seed 42\
    --pretrained_model_name_or_path data/mochi\
    --cache_dir "data/.cache"\
    --data_json_path "data/Merge-30k-Data/video2caption.json"\
    --validation_prompt_dir "data/validation_embeddings/validation_prompt_embed_mask"\
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
    --checkpointing_steps=500\
    --validation_steps 125\
    --validation_sampling_steps 8 \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --ema_decay 0.999\
    --log_validation\
    --output_dir="data/outputs/lq_euler_50_thresh0.05_multiphaseshedule_125-16,250-8,375-4,500-2"\
    --tracker_project_name PCM \
    --num_frames  163 \
    --scheduler_type pcm_linear_quadratic \
    --validation_guidance_scale "2.5,3.5,4.5" \
    --num_euler_timesteps 50 \
    --linear_quadratic_threshold 0.05 \
    --multi_phased_distill_schedule "125-16,250-8,375-4,500-2" \
    --use_ema

