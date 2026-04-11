#!/bin/bash
#SBATCH --job-name=wan_t2v_1.3B_finetune_qat_16
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=logs/wan_t2v_1.3B_finetune_qat_16.out
#SBATCH --error=logs/wan_t2v_1.3B_finetune_qat_16.err
#SBATCH --exclusive

source ~/conda/miniconda/bin/activate
conda activate matthew-fv

# Basic Info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# export TRITON_PRINT_AUTOTUNING=1  # to print the best config
export WANDB_API_KEY=YOUR_WANDB_API_KEY
MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
DATA_DIR=YOUR_DATA_DIR
VALIDATION_DATASET_FILE="examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json"
NUM_GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
# export CUDA_VISIBLE_DEVICES=4,5

# Training arguments
training_args=(
  --tracker_project_name "wan_t2v_finetune_qat"
  --output_dir "checkpoints/wan_14B_t2v_finetune_qat"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 768 
  --num_width 1280
  --num_frames 77
  --enable_gradient_checkpointing_type "full" # if OOM enable this
)

# Parallel arguments
parallel_args=(
  --num_gpus $TOTAL_GPUS 
  --sp_size 4
  --tp_size 1
  --hsdp_replicate_dim 4
  --hsdp_shard_dim 8
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
  # --log_validation 
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 200
  --validation_sampling_steps "50" 
  --validation_guidance_scale "5.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 200
  --training_state_checkpointing_steps 200
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
  --flow_shift 5
  --seed 1000
)

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS_PER_NODE \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
