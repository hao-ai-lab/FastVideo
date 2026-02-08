#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="Davids048/LTX2-Base-Diffusers"
# Also can use simple 1 video for overfitting experiments.
# DATA_DIR="/home/hal-jundas/codes/FastVideo/data/crush-smol"
DATA_DIR="<PATH_TO_PROCESSED_DATASET>"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
echo  VALIDATION_DATASET_FILE: $VALIDATION_DATASET_FILE
NUM_GPUS=4
OVERFIT_HEIGHT=480
OVERFIT_WIDTH=832
OVERFIT_FRAMES=73

training_args=(
  --tracker_project_name "ltx2_t2v_finetune"
  --output_dir "checkpoints/ltx2_t2v_finetune"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 10
  --num_height $OVERFIT_HEIGHT
  --num_width $OVERFIT_WIDTH
  --num_frames $OVERFIT_FRAMES
  --ltx2-first-frame-conditioning-p 0.1
  --enable_gradient_checkpointing_type "full"
  --mode "finetuning"
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
  --validation_steps 50
  --validation_sampling_steps "50"
  --validation_guidance_scale "3.0"
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
  --weight_decay 1e-4
  --max_grad_norm 1.0
  --lr_scheduler "linear"
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --dit_precision "fp32"
  --dit_cpu_offload False
  --dit_layerwise_offload False
  --text_encoder_cpu_offload False
  --image_encoder_cpu_offload False
  --vae_cpu_offload False
)

# NOTE: Setting this environment variable to TORCH_SDPA to avoid the issue of 
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA


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
