#!/bin/bash

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=false

RUN_NAME=$(date +"%m%d_%H%M")
echo "RUN_NAME: $RUN_NAME"

# Model paths for Self-Forcing DMD distillation.
GENERATOR_MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers"
REAL_SCORE_MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"  # Teacher
FAKE_SCORE_MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"  # Critic

DATA_DIR="data/matrixgame2"
VALIDATION_DATASET_FILE="examples/distill/MatrixGame2.0/validation.json"
NUM_GPUS=1

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/matrixgame2_self_forcing_distillation_pipeline.py \
    --config examples/distill/MatrixGame2.0/distill_dmd.yaml \
    --model_path "$GENERATOR_MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.distillation.real_score_model_path "$REAL_SCORE_MODEL_PATH" \
    --training.distillation.fake_score_model_path "$FAKE_SCORE_MODEL_PATH" \
    --training.data.data_path "$DATA_DIR" \
    --training.checkpoint.output_dir "checkpoints/matrixgame2_sf_${RUN_NAME}" \
    --training.tracker.run_name "${RUN_NAME}_test" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"