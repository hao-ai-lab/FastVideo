#!/bin/bash
#SBATCH --job-name=wg-ode
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=ode_train_output/ode_%j.out
#SBATCH --error=ode_train_output/ode_%j.err
#SBATCH --exclusive

set -e -x

# Environment Setup
source ~/conda/miniconda/bin/activate /mnt/weka/home/hao.zhang/conda/miniconda/envs/mhuo-fv
export PYTHONPATH="/mnt/weka/home/hao.zhang/kaiqin/FastVideo:$PYTHONPATH"

# Basic Info
export WANDB_API_KEY="7ff8b6e8356924f7a6dd51a0342dd1a422ea9352"
export WANDB_MODE=online
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
DATA_DIR="../traindata_0209_1500/ode_init/preprocessed"
VALIDATION_DATASET_FILE="examples/training/consistency_finetune/causal_wangame_ode_init/validation.json"
CKPT_SAFETENSOR="/mnt/weka/home/hao.zhang/mhuo/FastVideo/wangame_1.3b_with_warmup_lr_1e-5/checkpoint-7000/transformer/diffusion_pytorch_model.safetensors"

# Training arguments
training_args=(
  --tracker_project_name "wangame_ode_init"
  --output_dir "checkpoints/wangame_ode_init"
  --override_transformer_cls_name "CausalWanGameActionTransformer3DModel"
  --wandb_run_name "wangame_ode_init_64gpu"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 352
  --num_width 640
  --num_frames 81
  --warp_denoising_step
  --enable_gradient_checkpointing_type "full"
  --init_weights_from_safetensors $CKPT_SAFETENSOR
)

# Parallel arguments
parallel_args=(
  --num_gpus 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 64
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
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 6e-6
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
)

mkdir -p ode_train_output

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 8 \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wangame_ode_causal_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
