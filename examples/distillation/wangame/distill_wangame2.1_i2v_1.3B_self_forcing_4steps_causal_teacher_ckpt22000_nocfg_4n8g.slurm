#!/bin/bash
#SBATCH --job-name=wangame_self_forcing_causal
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=slurm_outputs/wangame_self_forcing_causal_teacher22000_4n8g/wangame_self_forcing_causal_%j.out
#SBATCH --error=slurm_outputs/wangame_self_forcing_causal_teacher22000_4n8g/wangame_self_forcing_causal_%j.err
#SBATCH --exclusive

set -euo pipefail

# ---- Env (mirror legacy WanGame slurm scripts) ----
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}

# Triton cache (avoid cross-job collisions; per-node task is OK in practice)
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}}

# Rendezvous (torchrun)
export MASTER_PORT=${MASTER_PORT:-29501}
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR=${MASTER_ADDR:-${nodes[0]}}

# W&B (recommended to pass WANDB_API_KEY via environment / secret manager)
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
export WANDB_MODE=${WANDB_MODE:-online}

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY in env." >&2
  exit 1
fi

source ~/conda/miniconda/bin/activate
conda activate alexfv

CONFIG=${CONFIG:-"examples/distillation/wangame/distill_wangame2.1_i2v_1.3B_self_forcing_4steps_causal_teacher_ckpt22000_nocfg_4n8g.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/wangame_self_forcing_causal_teacher22000_4n8g/${SLURM_JOB_ID}"}

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing distillation YAML config at: $CONFIG" >&2
  exit 1
fi

# 4 nodes × 8 GPUs
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=8

srun torchrun \
  --nnodes "$NUM_NODES" \
  --nproc_per_node "$GPUS_PER_NODE" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  --node_rank "$SLURM_PROCID" \
  fastvideo/training/distillation.py \
  --config "$CONFIG" \
  --override-output-dir "$OUTPUT_DIR"
