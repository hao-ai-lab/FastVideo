#!/bin/bash
#SBATCH --job-name=t2v
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_Wan2.2/t2v_g2e5_f1e5_%j.out
#SBATCH --error=dmd_Wan2.2/t2v_g2e5_f1e5_%j.err
#SBATCH --exclusive
set -e -x



export SLURM_JOB_NUM_NODES=1
export NODE_RANK=0
export SLURM_PROCID=0



# Environment Setup
# source ~/conda/miniconda/bin/activate
# conda activate your_env

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
# nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
# export MASTER_ADDR=${nodes[0]}
export MASTER_ADDR=$(hostname)
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE="online"
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export WANDB_API_KEY=your_wandb_api_key
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA


echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=2
MODEL_PATH="Davids048/LTX2-Base-Diffusers"
REAL_SCORE_MODEL_PATH="Davids048/LTX2-Base-Diffusers"
FAKE_SCORE_MODEL_PATH="Davids048/LTX2-Base-Diffusers"
DATA_DIR=data/test-text-preprocessing/single_node/
VALIDATION_DIR=your_validation_path  #(example:validation_64.json)
OUTPUT_DIR="checkpoints/ltx2_distillation"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
#   --tracker_project_name LTX2_distillation
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
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 2
)

# Validation arguments
# validation_args=(
#   --log_validation
#   --validation_dataset_file "$VALIDATION_DIR"
#   --validation_steps 200
#   --validation_sampling_steps "3"
#   --validation_guidance_scale "6.0" # not used for dmd inference
# )

# Optimizer arguments
optimizer_args=(
  --learning_rate 4e-6
  --lr_scheduler "cosine_with_min_lr"
  --min_lr_ratio 0.5
  --lr_warmup_steps 100
  --fake_score_learning_rate 2e-6
  --fake_score_lr_scheduler "cosine_with_min_lr"
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 200
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
  --flow_shift 5
  --seed 1000
)

# DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,757,522'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3
  --simulate_generator_forward 
  --log_visualization # disable if oom
)

# srun torchrun \
torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/ltx2_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"