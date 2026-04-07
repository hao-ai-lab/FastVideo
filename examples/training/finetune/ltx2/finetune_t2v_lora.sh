#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="FastVideo/LTX2-Distilled-Diffusers"
DATA_DIR="data/crush-smol"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=1

training_args=(
  --tracker_project_name "ltx2_t2v_lora_finetune"
  --output_dir "checkpoints/ltx2_t2v_lora_finetune"
  --max_train_steps 2000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 8
  --num_latent_t 10
  --num_height 480
  --num_width 832
  --num_frames 77
  --ltx2-first-frame-conditioning-p 0.1
  --enable_gradient_checkpointing_type "full"
)

parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size $NUM_GPUS
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

dataset_args=(
  --data_path $DATA_DIR
  --dataloader_num_workers 1
)

validation_args=(
  --log_validation
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "3.0"
)

optimizer_args=(
  --learning_rate 2e-4
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
  --weight_decay 1e-4
  --max_grad_norm 1.0
  --lora_training True
  --lora_rank 16
  --lora_alpha 16
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
)

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/ltx2_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
