export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset
# DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Vchitect-2M-laten-93x512x512/val/
VALIDATION_DIR=mini_i2v_dataset/crush-smol_raw/validation.json
NUM_GPUS=2
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]
export TOKENIZERS_PARALLELISM=false
CHECKPOINT_PATH="outputs_train_test/wan_finetune/checkpoint-10"

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/v1/training/wan_distillation_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_dataset_file  "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 8 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim $NUM_GPUS  \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 5 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 10 \
    --validation_steps 3 \
    --validation_sampling_steps "3" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "outputs_dmd/wan_finetune" \
    --tracker_project_name Wan_distillation \
    --num_height 448 \
    --num_width 832 \
    --num_frames  29 \
    --flow_shift 8 \
    --validation_guidance_scale "1.0" \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --vae_precision "bf16" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --student_critic_update_ratio 3 \
    --denoising_step_list '1000,757,522' \
    --min_step_ratio 0.02 \
    --max_step_ratio 0.98 \
    --teacher_guidance_scale 3.5 \
    --enable_gradient_checkpointing_type "full" \
    --seed 1024 \

# validation_preprocessed_path