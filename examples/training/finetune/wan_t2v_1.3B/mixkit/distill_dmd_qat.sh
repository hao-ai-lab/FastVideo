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
# Data: run examples/datasets/mixkit/download_dataset.sh first.
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
    --config examples/training/finetune/wan_t2v_1.3B/mixkit/distill_dmd_qat.yaml \
    --model_path "${BASE}" \
    --engine.num_gpus "${NUM_GPUS}" \
    --engine.parallelism.hsdp_replicate_dim "${NUM_GPUS}" \
    --training.distillation.real_score_model_path "${BASE}" \
    --training.distillation.fake_score_model_path "${BASE}" \
    --pipeline.components.transformer_weights "${INIT_WEIGHTS}" \
    --training.data.data_path "${DATA_DIR}" \
    --training.validation.dataset_file "${VALIDATION_FILE}"
