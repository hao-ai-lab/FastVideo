export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY='your_wandb_api_key'
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=~/train/
VALIDATION_DATASET_FILE=[your validation dataset file]
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

CHECKPOINT_PATH="$DATA_DIR/outputs/wan_finetune/checkpoint-5"

# Make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_training_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_dataset_file "$VALIDATION_DATASET_FILE" \
    --train_batch_size 1 \
    --num_latent_t 16 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim $NUM_GPUS \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 30000 \
    --learning_rate 1e-6 \
    --mixed_precision "bf16" \
    --training_state_checkpointing_steps 6000 \
    --validation_steps 100 \
    --validation_sampling_steps "50" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "$DATA_DIR/outputs/wan_finetune" \
    --tracker_project_name VSA_finetune \
    --num_height 448 \
    --num_width 832 \
    --num_frames  61 \
    --flow_shift 3 \
    --validation_guidance_scale "5.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --VSA_decay_sparsity 0.9 \
    --VSA_decay_rate 0.03 \
    --VSA_decay_interval_steps 30 \
    --VSA_val_sparsity 0.9 \
    --enable_gradient_checkpointing_type "full"
# --resume_from_checkpoint "$CHECKPOINT_PATH"
