#!/bin/bash

export WANDB_API_KEY="7ff8b6e8356924f7a6dd51a0342dd1a422ea9352"
export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_PATH="weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers"
DATA_DIR="../traindata_0205_1330/data/0_static_plus_w_only/preprocessed"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# IP=[MASTER NODE IP]

source ~/conda/miniconda/bin/activate
conda activate /mnt/weka/home/hao.zhang/conda/miniconda/envs/mhuo-fv
export PYTHONPATH="/mnt/weka/home/hao.zhang/kaiqin/FastVideo:$PYTHONPATH"

# Training arguments
training_args=(
  --override-pipeline-cls-name "WanLingBotImageToVideoPipeline"
  --override-transformer-cls-name "WanLingBotTransformer3DModel"
  --tracker_project_name "wangame_lingbot_test"
  --output_dir "wangame_lingbot_test"
  --max_train_steps 100
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 352
  --num_width 640
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 1
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "40"
  --validation_guidance_scale "1.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/wangame_lingbot_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"