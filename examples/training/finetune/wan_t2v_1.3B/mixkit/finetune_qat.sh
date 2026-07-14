#!/bin/bash
# QAD recipe — quantization-aware finetune of Wan2.1-T2V-1.3B with fake-quant
# (Attn-QAT) attention.
#
# The 4-bit attention path is selected purely by env var (config-driven, no
# monkey-patching): FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN routes attention
# through the fake-quantized Triton kernel (straight-through estimator), so the
# DiT learns to absorb FP4 attention error instead of fighting it.
#
# Data: run download_mixkit_data.sh first (preprocessed Parquet).
#
# Verified end-to-end on Blackwell (GB200/sm_100): the ATTN_QAT_TRAIN backend is
# selected (not a fallback), forward+backward run, loss/grad are healthy, and
# validation generates videos. The kernel is Triton so it runs on sm_100 and
# sm_120 alike (the FP4 inference kernel, by contrast, is sm_120-only).
set -euo pipefail

export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN   # <-- enables Attn-QAT training
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR=${1:-"data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset/"}
VALIDATION_FILE="$(dirname "$0")/../crush_smol/validation.json"
NUM_GPUS=${NUM_GPUS:-4}

torchrun --nnodes 1 --nproc_per_node "${NUM_GPUS}" \
    fastvideo/training/wan_training_pipeline.py \
    --num_gpus "${NUM_GPUS}" --sp_size "${NUM_GPUS}" --tp_size 1 \
    --hsdp_replicate_dim 1 --hsdp_shard_dim "${NUM_GPUS}" \
    --model_path "${MODEL_PATH}" --pretrained_model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_DIR}" --dataloader_num_workers 1 \
    --max_train_steps 2000 --train_batch_size 1 --train_sp_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_latent_t 20 --num_height 480 --num_width 832 --num_frames 77 \
    --enable_gradient_checkpointing_type full \
    --log_validation --validation_dataset_file "${VALIDATION_FILE}" \
    --validation_steps 200 --validation_sampling_steps 50 --validation_guidance_scale 3.0 \
    --learning_rate 5e-5 --mixed_precision bf16 --weight_decay 1e-4 --max_grad_norm 1.0 \
    --weight_only_checkpointing_steps 500 --training_state_checkpointing_steps 500 \
    --tracker_project_name wan_t2v_qat_finetune --output_dir checkpoints/wan_t2v_qat_finetune \
    --inference_mode False --training_cfg_rate 0.1 --not_apply_cfg_solver \
    --dit_precision fp32 --num_euler_timesteps 50 --ema_start_step 0 \
    --multi_phased_distill_schedule "4000-1"
