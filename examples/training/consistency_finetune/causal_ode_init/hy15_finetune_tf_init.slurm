#!/bin/bash
#SBATCH --job-name=hy15_tf_3333
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=hy15_tf_3333_output/hy15_tf_3333_%j.out
#SBATCH --error=hy15_tf_3333_output/hy15_tf_3333_%j.err
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
# export WANDB_API_KEY="8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde"
export WANDB_API_KEY="50632ebd88ffd970521cec9ab4a1a2d7e85bfc45"
# export WANDB_API_KEY='your_wandb_api_key_here'
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

source ~/conda/miniconda/bin/activate
conda activate wei-fv-distill
export HOME="/mnt/weka/home/hao.zhang/wei"

# Configs
NUM_GPUS=8
NUM_TOTAL_GPUS=64

MODEL_PATH="weizhou03/SFHunyuanVideo-1.5-Diffusers-480p_t2v"
DATA_DIR="/mnt/weka/home/hao.zhang/wl/sharefs/wl/release/fv-ode-preprocessing-16k-hy15-121"
VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wl/FastVideo/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/validation_64.json"
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name "hy15_ode_init"
  --output_dir "/mnt/weka/home/hao.zhang/wei/hy15_worldplay_tf_init_3333"
  --wandb_run_name "hy15_worldplay_tf_init_3333"
  --warp_denoising_step
  --dmd_denoising_steps '1000,760,520,280,0'
  --log_visualization
  --visualization_steps 100
  --max_train_steps 30000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 848
  --num_frames 81
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS # 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1 # 64
  --hsdp_shard_dim $NUM_TOTAL_GPUS
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --text-encoder-cpu-offload
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR" 
  --dataloader_num_workers 4
)

# Optimizer arguments
optimizer_args=(
  --optimizer-type "muon"
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 400
  --weight_decay 0.01
  --max_grad_norm 1.0
)

validation_args=(
  # --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 10
  --validation_sampling_steps "4"
  --validation_guidance_scale "1.0" # not used for dmd inference
  # --vae_cpu_offload True
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 6
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
  # --enable_gradient_checkpointing_type "full"
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/hy15_ode_causal_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"