#!/bin/bash
# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate fastvideo

# Basic Info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=localhost
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=VMOBA_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=2
MODEL_PATH="huggingface/models/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="huggingface/datasets/Wan-Syn_77x448x832_600k/train_Part_100/train"
VMOBA_CONFIG="fastvideo/configs/backend/vmoba/wan_1.3B_77_448_832_train.json"
VALIDATION_DATASET_FILE="huggingface/datasets/Mixkit-Src/validation_16.json"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name wan_t2v_VMOBA
  --output_dir "checkpoints/wan_t2v_finetune_VMOBA"
  --max_train_steps 1000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 8
  --num_latent_t 16
  --num_height 448
  --num_width 832
  --num_frames 61
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
parallel_args=(
  --num_gpus 2
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 2
  --hsdp_shard_dim 1
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 100
  --validation_sampling_steps "50"
  --validation_guidance_scale "5.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-6
  --mixed_precision "bf16"
  --checkpointing_steps 1000
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 3
  --seed 1000
)

# VMoba arguments
vmoba_args=(
  --moba-config-path "$VMOBA_CONFIG"
)


torchrun \
--nnodes 1 \
--nproc_per_node $NUM_GPUS \
--node_rank 0 \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${vmoba_args[@]}"  