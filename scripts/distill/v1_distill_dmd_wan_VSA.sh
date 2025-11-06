export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY=
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset/
VALIDATION_DIR=mini_i2v_dataset/crush-smol_raw/validation.json
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export TOKENIZERS_PARALLELISM=false

# Train generator with VSA
# Make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_distillation_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --real_score_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --fake_score_model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_dataset_file  "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 16 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim $NUM_GPUS  \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 30000 \
    --learning_rate 2e-6 \
    --mixed_precision "bf16" \
    --training_state_checkpointing_steps 400 \
    --validation_steps 100 \
    --validation_sampling_steps "3" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "outputs_dmd/wan_finetune" \
    --tracker_project_name Wan_distillation \
    --num_height 448 \
    --num_width 832 \
    --num_frames 61 \
    --flow_shift 8 \
    --validation_guidance_scale "6.0" \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --vae_precision "bf16" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --generator_update_interval 5 \
    --dmd_denoising_steps '1000,757,522' \
    --min_timestep_ratio 0.02 \
    --max_timestep_ratio 0.98 \
    --real_score_guidance_scale 3.5 \
    --seed 1024 \
    --VSA_sparsity 0.8 