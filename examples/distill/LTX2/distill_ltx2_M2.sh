#!/bin/bash
#SBATCH --job-name=ltx2_distillation
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=logs/ltx2_distillation.out
#SBATCH --error=logs/ltx2_distillation.err
#SBATCH --exclusive
set -e -x

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
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

export WANDB_API_KEY=50632ebd88ffd970521cec9ab4a1a2d7e85bfc45
# Configs
MODEL_PATH="FastVideo/LTX2-Distilled-Diffusers"
REAL_SCORE_MODEL_PATH="Davids048/LTX2-Base-Diffusers"
FAKE_SCORE_MODEL_PATH="Davids048/LTX2-Base-Diffusers"
DATA_DIR=/mnt/weka/home/hao.zhang/data/FastVideo/LTX2_text_encoding_dataset
VALIDATION_DIR="examples/distill/LTX2/validation.json"
NUM_GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
OUTPUT_DIR="checkpoints/ltx2_distillation"
# export CUDA_VISIBLE_DEVICES=4,5

# Training arguments
training_args=(
  --tracker_project_name LTX2_distillation
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 31
  --num_height 1088 
  --num_width 1920
  --num_frames 121
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
parallel_args=(
  --num_gpus $TOTAL_GPUS 
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 8
  --hsdp_shard_dim 8
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation 
  --validation_dataset_file "$VALIDATION_DIR"
  --validation_steps 5
  --validation_sampling_steps "8" 
  --validation_guidance_scale "1.0" # used by validation inference; keep aligned with basic_ltx2_distilled defaults
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 5  # TODO: need to determine the correct value
  --seed 1000
)

# DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,993,987,981,975,909,725,421'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3
  --simulate_generator_forward 
  --log_visualization # disable if oom
  --VSA_sparsity 0.8
)

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS_PER_NODE \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/ltx2_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"
