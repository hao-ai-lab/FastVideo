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

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
srun torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc_per_node $NUM_GPUS \
  --node_rank $SLURM_PROCID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/ode_causal_pipeline.py \
    --config examples/training/consistency_finetune/causal_ode_init/finetune_ode_init_16k_vidprom.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.hsdp_replicate_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.checkpoint.output_dir "$OUTPUT_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE" \
    --pipeline.components.transformer_weights "$INIT_WEIGHTS_FROM_SAFETENSORS"
