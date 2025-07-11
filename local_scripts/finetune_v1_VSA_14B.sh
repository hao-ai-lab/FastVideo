export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn_77x768x1280/latents_i2v/train/
VALIDATION_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn_77x768x1280/latents_i2v/test_8/
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

CHECKPOINT_PATH="$DATA_DIR/outputs/wan_finetune/checkpoint-5"

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/v1/training/wan_training_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_preprocessed_path "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 20 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim 1 \
    --hsdp-shard-dim 8 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 30000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 200 \
    --validation_steps 100 \
    --validation_sampling_steps "2" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --training_cfg_rate 0.1 \
    --output_dir "outputs_VSA/wan_finetune" \
    --tracker_project_name VSA_finetune \
    --num_height 768 \
    --num_width 1280 \
    --num_frames 77 \
    --flow_shift 1 \
    --validation_guidance_scale "5.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --VSA_decay_rate 0.03 \
    --VSA_decay_interval_steps 20 \
    --VSA_sparsity 0.0 \
    --enable_gradient_checkpointing_type "full"
# --resume_from_checkpoint "$CHECKPOINT_PATH"
