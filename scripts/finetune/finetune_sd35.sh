#!/usr/bin/env bash
# Fine-tune Stable Diffusion 3.5 Medium.
# Requires preprocessed parquet data from v1_preprocess_sd35_data.sh.
#
# Usage:
#   DATA_DIR=data/sd35_dataset_preprocessed \
#   VALIDATION_DATASET_FILE=examples/training/finetune/sd35/validation.json \
#   bash scripts/finetune/finetune_sd35.sh

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

DATA_DIR=${DATA_DIR:-"[your data dir]"}
VALIDATION_DATASET_FILE=${VALIDATION_DATASET_FILE:-"[your validation dataset file]"}
NUM_GPUS=${NUM_GPUS:-1}

torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/training/sd35_training_pipeline.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3.5-medium \
    --inference_mode False \
    --data_path "$DATA_DIR" \
    --validation_dataset_file "$VALIDATION_DATASET_FILE" \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --tp_size 1 \
    --hsdp_replicate_dim 1 \
    --hsdp_shard_dim $NUM_GPUS \
    --num_gpus $NUM_GPUS \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 5000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --training_state_checkpointing_steps 1000 \
    --validation_steps 500 \
    --validation_sampling_steps "28" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --training_cfg_rate 0.0 \
    --output_dir="$DATA_DIR/outputs/sd35_finetune" \
    --tracker_project_name sd35_finetune \
    --num_height 512 \
    --num_width 512 \
    --num_frames 1 \
    --validation_guidance_scale "6.0" \
    --num_euler_timesteps 28 \
    --weight_decay 0.01 \
    --not_apply_cfg_solver \
    --dit_precision "fp32" \
    --max_grad_norm 1.0 \
    --weighting_scheme "logit_normal" \
    --logit_mean 0.0 \
    --logit_std 1.0 \
    --enable_gradient_checkpointing_type "full"
