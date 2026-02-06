#!/bin/bash
#SBATCH --job-name=wangame_1.3b
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=wangame_1.3b_output/wangame_1.3b_%j.out
#SBATCH --error=wangame_1.3b_output/wangame_1.3b_%j.err
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG_SUBSYS=INIT,NET
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="d5b02b05e30d8cb34c7b31c6ae10416fc26dcb66"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

source ~/conda/miniconda/bin/activate
conda activate mhuo-fv
export HOME="/mnt/weka/home/hao.zhang/mhuo"

# Configs
NUM_GPUS=8
NUM_NODES=2
NUM_TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))
BS_PER_GPU=2
GRADIENT_ACCUMULATION_STEPS=1
WANDB_RUN_NAME="MC_WASD_bs32_with_warmup_lr_1e-4"
FREEZE_DIT=False
RUN_DIR="wangame_1.3b_with_warmup_lr_1e-4"
CHECKPOINTING_STEPS=1000
SCENE_PROMPT=""
ACTION_WARMUP_STEPS=200
LEARNING_RATE=1e-4

MODEL_PATH="weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers"
# MC w only 
DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/1_wasd_only/preprocessed"
VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_wsad.json"

# DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0204_2130/preprocessed"
# VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_mc.json"
# DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0204_1600/preprocessed"
# VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_doom.json"
# DATA_DIR="mc_wasd_10/preprocessed/combined_parquet_dataset"
# VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_overfit.json"


# Training arguments
training_args=(
  --tracker_project_name "wangame_1.3b"
  --output_dir $RUN_DIR
  --wandb_run_name "$WANDB_RUN_NAME"
  --max_train_steps 10000
  --train_batch_size $BS_PER_GPU
  --train_sp_batch_size $BS_PER_GPU
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
  --num_latent_t 20
  --num_height 352
  --num_width 640
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --train_action_only $FREEZE_DIT
  --action_warmup_steps $ACTION_WARMUP_STEPS
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_TOTAL_GPUS
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
  --validation_num_samples $NUM_TOTAL_GPUS
)

# Optimizer arguments
optimizer_args=(
  --learning_rate $LEARNING_RATE
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 100000
  --training_state_checkpointing_steps $CHECKPOINTING_STEPS
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 2
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t

if [ $NUM_NODES -eq 1 ]; then
  torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    fastvideo/training/wangame_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
else
  srun torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    fastvideo/training/wangame_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
fi