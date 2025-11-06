export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

DATA_DIR=[your data dir]
VALIDATION_DATASET_FILE=[your validation dataset file]
NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS\
    fastvideo/training/wan_training_pipeline.py\
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --data_path "$DATA_DIR"\
    --validation_dataset_file "$VALIDATION_DATASET_FILE"\
    --train_batch_size=4 \
    --num_latent_t 20 \
    --sp_size 4 \
    --tp_size 1 \
    --hsdp_replicate_dim 1 \
    --hsdp_shard_dim 4 \
    --num_gpus $NUM_GPUS \
    --train_sp_batch_size 1\
    --dataloader_num_workers 10\
    --gradient_accumulation_steps=1 \
    --max_train_steps=5000 \
    --learning_rate=1e-6\
    --mixed_precision="bf16"\
    --training_state_checkpointing_steps 6000 \
    --validation_steps 200\
    --validation_sampling_steps "2,4,8" \
    --log_validation \
    --checkpoints_total_limit 3\
    --ema_start_step 0\
    --training_cfg_rate 0.0\
    --output_dir="$DATA_DIR/outputs/wan_finetune"\
    --tracker_project_name wan_finetune \
    --num_height 480 \
    --num_width 832 \
    --num_frames  81 \
    --validation_guidance_scale "6.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --weight_decay 0.01 \
    --not_apply_cfg_solver \
    --dit_precision "fp32" \
    --max_grad_norm 1.0 \
    --enable_gradient_checkpointing_type "full"