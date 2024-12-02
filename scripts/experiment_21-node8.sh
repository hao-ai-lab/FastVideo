export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_DIR="$HOME"
export WANDB_MODE=offline
export WANDB_API_KEY=4f6de3765d6464f43e0506ec7d785641af645e73
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_PROTO=simple

DATA_DIR=/data
IP=10.4.139.86
CACHE_DIR=/data/.cache
EXPERIMENT=4step_infer_lq_euler_50_thresh0.1_lrg_0.75
OUTPUT_DIR=$DATA_DIR/outputs/$EXPERIMENT
export WANDB_DIR=$DATA_DIR/wandb/
torchrun --nnodes 8 --nproc_per_node 8\
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$IP:29500 \
    fastvideo/distill.py\
    --seed 42\
    --pretrained_model_name_or_path $DATA_DIR/mochi\
    --cache_dir $CACHE_DIR \
    --data_json_path "$DATA_DIR/Merge-30k-Data/video2caption.json"\
    --validation_prompt_dir "$DATA_DIR/validation_embeddings/validation_prompt_embed_mask"\
    --gradient_checkpointing\
    --train_batch_size=1\
    --num_latent_t 24\
    --sp_size 8\
    --train_sp_batch_size 1\
    --dataloader_num_workers 4\
    --gradient_accumulation_steps=1\
    --max_train_steps=2000\
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --checkpointing_steps=500\
    --validation_steps 125\
    --validation_sampling_steps 4 \
    --checkpoints_total_limit 3\
    --allow_tf32\
    --ema_start_step 0\
    --cfg 0.0\
    --ema_decay 0.999\
    --log_validation\
    --output_dir=$OUTPUT_DIR \
    --tracker_project_name PCM \
    --num_frames  139 \
    --scheduler_type pcm_linear_quadratic \
    --validation_guidance_scale "1.5,2.5,4.5,6.5" \
    --num_euler_timesteps 50 \
    --linear_quadratic_threshold 0.1 \
    --linear_range 0.75 

