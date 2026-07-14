#!/bin/bash
# QAD recipe stage 2 — quantization-aware DMD distillation of Wan2.1-T2V-1.3B
# down to 3 sampling steps, with the GENERATOR in fake-quant Attn-QAT and the
# teacher (real_score) + critic (fake_score) at full precision.
#
# Generator-only QAT is config-driven: FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN
# is applied to the generator only, because the loader masks it (and the
# nvfp4_qat quant) for the teacher/critic via the `_loading_teacher_critic_model`
# flag (see fastvideo/models/loader/component_loader.py). No monkey-patching.
#
# Init the generator from the stage-1 finetune checkpoint (finetune_qat.sh).
# Data: run download_mixkit_data.sh first.
#
# Verified end-to-end on Blackwell (GB200/sm_100): generator loads with
# ATTN_QAT_TRAIN while teacher/critic load full-precision; the DMD double loop
# runs (generator updates every generator_update_interval steps, critic every
# step), 3-step validation generates videos, checkpoint saved.
set -euo pipefail

export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN   # generator-only (loader-gated)
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false

BASE="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR=${1:-"data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset/"}
# Generator init weights = the stage-1 QAT-finetune checkpoint.
INIT_WEIGHTS=${2:-"checkpoints/wan_t2v_qat_finetune/checkpoint-2000/transformer/diffusion_pytorch_model.safetensors"}
VALIDATION_FILE="$(dirname "$0")/../crush_smol/validation.json"
NUM_GPUS=${NUM_GPUS:-4}

torchrun --nnodes 1 --nproc_per_node "${NUM_GPUS}" \
    fastvideo/training/wan_distillation_pipeline.py \
    --num_gpus "${NUM_GPUS}" --sp_size 1 --tp_size 1 \
    --hsdp_replicate_dim "${NUM_GPUS}" --hsdp_shard_dim 1 \
    --model_path "${BASE}" --pretrained_model_name_or_path "${BASE}" \
    --real_score_model_path "${BASE}" --fake_score_model_path "${BASE}" \
    --init_weights_from_safetensors "${INIT_WEIGHTS}" \
    --data_path "${DATA_DIR}" --dataloader_num_workers 4 \
    --max_train_steps 2000 --train_batch_size 1 --train_sp_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_latent_t 20 --num_height 480 --num_width 832 --num_frames 77 \
    --enable_gradient_checkpointing_type full \
    --log_validation --validation_dataset_file "${VALIDATION_FILE}" \
    --validation_steps 200 --validation_sampling_steps 3 --validation_guidance_scale 6.0 \
    --learning_rate 2e-6 --mixed_precision bf16 --weight_decay 0.01 --max_grad_norm 1.0 \
    --weight_only_checkpointing_steps 500 --training_state_checkpointing_steps 500 \
    --tracker_project_name wan_t2v_distill_dmd_qat \
    --output_dir checkpoints/wan_t2v_distill_dmd_qat \
    --inference_mode False --dit_precision fp32 --ema_start_step 0 --training_cfg_rate 0.0 \
    --generator_update_interval 5 --real_score_guidance_scale 2.0 \
    --dmd_denoising_steps '1000,757,522' --min_timestep_ratio 0.02 --max_timestep_ratio 0.98
