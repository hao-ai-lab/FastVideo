#!/bin/bash
#SBATCH --job-name=2e6B8_16kFV_ode_vidprom
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=ode_vidprom16k/ode_vidprom8b16k_2e-6.out
#SBATCH --error=ode_vidprom16k/ode_vidprom8b16k_2e-6.err
#SBATCH --exclusive
set -e -x

# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate your-conda-env

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
export WANDB_API_KEY=your-wandb-api-key
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"


MODEL_PATH="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
DATA_DIR="your-data-dir"
VALIDATION_DATASET_FILE="examples/training/consistency_finetune/causal_ode_init/validation.json"
OUTPUT_DIR="your-output-dir"
INIT_WEIGHTS_FROM_SAFETENSORS="your-init-weights-from-safetensors" # bidirectional weights from Wan2.1-T2V-1.3B-Diffusers
NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name "wan_ode_init"
  --output_dir $OUTPUT_DIR
  --override_transformer_cls_name "CausalWanTransformer3DModel"
  --wandb_run_name "vidprom_8b16k_ode_init_2e-6"
  # --resume_from_checkpoint "ode_init_diffusers/"
  --warp_denoising_step
  --log_visualization
  --max_train_steps 6001
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --num_frames 81
  --dmd_denoising_steps "1000,750,500,250"
  --enable_gradient_checkpointing_type "full"
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim $NUM_GPUS
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
  --dataloader_num_workers 1
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 50
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
  --init_weights_from_safetensors $INIT_WEIGHTS_FROM_SAFETENSORS
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 500
  --training_state_checkpointing_steps 500
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
  # --enable_gradient_checkpointing_type "full"
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
srun torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc_per_node $NUM_GPUS \
  --node_rank $SLURM_PROCID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/ode_causal_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
