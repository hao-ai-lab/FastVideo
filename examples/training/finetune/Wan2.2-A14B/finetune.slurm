#!/bin/bash
#SBATCH --job-name=moe_finetune
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=moe_output/moe_%j.out
#SBATCH --error=moe_output/moe_%j.err
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
export WANDB_API_KEY="50632ebd88ffd970521cec9ab4a1a2d7e85bfc45"
# export WANDB_API_KEY='your_wandb_api_key_here'
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

source ~/conda/miniconda/bin/activate
conda activate wei-fv
export HOME="/mnt/weka/home/hao.zhang/wei"


# Configs
NUM_GPUS=8
NUM_TOTAL_GPUS=16

# Model paths for Self-Forcing DMD distillation with Wan2.2:
GENERATOR_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"  # Updated to Wan2.2
# REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers" # Teacher model
# FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers" # Critic model

# DATA_DIR="data/test-text-preprocessing/Node_0_GPU_1_File_1/combined_parquet_dataset/"
DATA_DIR="/mnt/weka/home/hao.zhang/wei/FastVideo/data/crush-smol_processed_t2v/combined_parquet_dataset"
# DATA_DIR=data/crush-smol_processed_t2v/combined_parquet_dataset
# DATA_DIR="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn-upload/latents_i2v/train/"
# VALIDATION_DATASET_FILE="data/crush-smol-single_processed_t2v/validation.json"
VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wei/FastVideo/data/crush-smol-single_processed_t2v/validation.json"

# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

training_args=(
  --tracker_project_name VSA_finetune  # Updated for Wan2.2
  --output_dir "/mnt/sharefs/users/hao.zhang/wei/Wan2.2-MoE-finetune"
  --max_train_steps 200
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 448  # Updated to match Wan2.2 config
  --num_width 832   # Updated to match Wan2.2 config
  --enable_gradient_checkpointing_type "full"
  # --log_visualization
  --num_frames 81
  # --init_weights_from_safetensors /mnt/sharefs/users/hao.zhang/wl/models/sf_ode_init_wan22_checkpoints/high/3k/
  # --init_weights_from_safetensors_2 /mnt/sharefs/users/hao.zhang/wl/models/sf_ode_init_wan22_checkpoints/low/3k/
)

parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS # 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1 # 64
  --hsdp_shard_dim $NUM_TOTAL_GPUS
)

model_args=(
  --model_path $GENERATOR_MODEL_PATH
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

validation_args=(
#   --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 50
  --validation_sampling_steps "40"
  --validation_guidance_scale "5.0" # not used for dmd inference
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --dit_cpu_offload True
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 5
  --seed 1000
)

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
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