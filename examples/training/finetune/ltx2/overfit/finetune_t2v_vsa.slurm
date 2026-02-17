#!/bin/bash
#SBATCH --job-name=ltx2_vsa_finetune
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --exclusive

# Create logs directory
mkdir -p slurm_logs

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG_SUBSYS=INIT,NET
export MASTER_PORT=29500

# Get master address
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$head_node

echo "Node list: $SLURM_JOB_NODELIST"
echo "Master Addr: $MASTER_ADDR"

# Activate environment
# Using the same environment as observed in other Slurm scripts in this repo (e.g., distill_ltx2.slurm)
# Uncomment the following lines if you need to activate a specific conda environment
source ~/conda/miniconda/bin/activate
conda activate matthew-fv  

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_TAGS="vsa,SFT"
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN

# Calculate absolute paths
REPO_ROOT="/mnt/weka/home/hao.zhang/ds8-agent/codes/FastVideo-demo"
SCRIPT_DIR="$REPO_ROOT/examples/training/finetune/ltx2/overfit"

echo "Script Dir: $SCRIPT_DIR"
echo "Repo Root: $REPO_ROOT"

# Configs
MODEL_PATH="FastVideo/LTX2-Distilled-Diffusers"
# Use absolute paths based on script location
DATA_DIR="$SCRIPT_DIR/data"
VALIDATION_DATASET_FILE="$SCRIPT_DIR/validation.json"

echo "DATA_DIR: $DATA_DIR"
echo "VALIDATION_DATASET_FILE: $VALIDATION_DATASET_FILE"

NUM_NODES=8
GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

OVERFIT_HEIGHT=1088
OVERFIT_WIDTH=1920
OVERFIT_FRAMES=121

training_args=(
  --tracker_project_name "ltx2_t2v_finetune"
  --output_dir "$SCRIPT_DIR/checkpoints/ltx2_t2v_finetune"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 2
  --num_latent_t 16
  --num_height $OVERFIT_HEIGHT
  --num_width $OVERFIT_WIDTH
  --num_frames $OVERFIT_FRAMES
  --ltx2-first-frame-conditioning-p 0.0
  --enable_gradient_checkpointing_type "full"
  --mode "finetuning"
)

parallel_args=(
  --num_gpus $TOTAL_GPUS
  --sp_size 2
  --tp_size 1
  --hsdp_replicate_dim $NUM_NODES
  --hsdp_shard_dim $GPUS_PER_NODE
)

model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

dataset_args=(
  --data_path $DATA_DIR
  --dataloader_num_workers 1
)

validation_args=(
  --log_validation
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 20
  --validation_sampling_steps "8"
  --validation_guidance_scale "1.0"
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
  --weight_decay 1e-4
  --max_grad_norm 1.0
  --lr_scheduler "linear"
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --dit_precision "fp32"
  --dit_cpu_offload False
  --dit_layerwise_offload False
  --text_encoder_cpu_offload False
  --image_encoder_cpu_offload False
  --vae_cpu_offload False
)

# VSA arguments
vsa_args=(
  --VSA_decay_rate 0.7 \
  --VSA_decay_interval_steps 1 \
  --VSA_sparsity 0.7 \
)

# Set PYTHONPATH to include the repo root
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

# Change into repo root so internal relative imports work if any
cd $REPO_ROOT

srun torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $SLURM_PROCID \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      fastvideo/training/ltx2_training_pipeline.py \
      "${parallel_args[@]}" \
      "${model_args[@]}" \
      "${dataset_args[@]}" \
      "${training_args[@]}" \
      "${optimizer_args[@]}" \
      "${validation_args[@]}" \
      "${miscellaneous_args[@]}" \
      "${vsa_args[@]}"
