#!/usr/bin/env bash
# Submit a multi-node Slurm training job.
#
# Usage:
#   bash examples/train/run_slurm.sh <config.yaml> <num_nodes> [--dotted.key value ...]
#
# Examples:
#   bash examples/train/run_slurm.sh examples/train/configs/example.yaml 2
#   bash examples/train/run_slurm.sh examples/train/configs/distill_wan2.1_t2v_1.3B_dmd2.yaml 4 \
#       --training.optimizer.learning_rate 1e-5
#   bash examples/train/run_slurm.sh examples/train/configs/example.yaml 8 \
#       --training.checkpoint.resume_from_checkpoint outputs/my_run/checkpoint-1000
#
# Environment variables (override defaults):
#   PARTITION       Slurm partition           (default: main)
#   NUM_GPUS        GPUs per node             (default: 8)
#   CPUS_PER_TASK   CPUs per task             (default: 128)
#   MEM             Memory per node           (default: 1440G)
#   JOB_NAME        Slurm job name            (default: derived from config)
#   OUTPUT_DIR      Directory for slurm logs  (default: slurm_logs)
#   MASTER_PORT     Rendezvous port           (default: 29500)
#   EXCLUDE         Nodes to exclude          (default: "")
#   WANDB_API_KEY   W&B API key               (default: "")
#   WANDB_MODE      W&B mode                  (default: online)

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> <num_nodes> [extra flags...]}"
NUM_NODES="${2:?Usage: $0 <config.yaml> <num_nodes> [extra flags...]}"
shift 2

# ── Defaults ──────────────────────────────────────────────────────
PARTITION="${PARTITION:-main}"
NUM_GPUS="${NUM_GPUS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
MEM="${MEM:-1440G}"
MASTER_PORT="${MASTER_PORT:-29500}"
EXCLUDE="${EXCLUDE:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

TOTAL_GPUS=$(( NUM_NODES * NUM_GPUS ))

CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
JOB_NAME="${JOB_NAME:-${CONFIG_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/slurm}"
mkdir -p "${OUTPUT_DIR}"

# ── Build sbatch args ─────────────────────────────────────────────
SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --nodes="${NUM_NODES}"
    --ntasks="${NUM_NODES}"
    --ntasks-per-node=1
    --gres="gpu:${NUM_GPUS}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --output="${OUTPUT_DIR}/${JOB_NAME}_%j.out"
    --error="${OUTPUT_DIR}/${JOB_NAME}_%j.err"
    --exclusive
)

if [[ -n "${EXCLUDE}" ]]; then
    SBATCH_ARGS+=(--exclude="${EXCLUDE}")
fi

# ── Collect extra overrides for the training script ───────────────
EXTRA_ARGS=("$@")

echo "=== Slurm Training Submission ==="
echo "Config:      ${CONFIG}"
echo "Nodes:       ${NUM_NODES}"
echo "GPUs/node:   ${NUM_GPUS}"
echo "Total GPUs:  ${TOTAL_GPUS}"
echo "Partition:   ${PARTITION}"
echo "Job name:    ${JOB_NAME}"
echo "Extra args:  ${EXTRA_ARGS[*]:-}"
echo "================================="

# ── Submit ────────────────────────────────────────────────────────
sbatch "${SBATCH_ARGS[@]}" <<EOF
#!/bin/bash
set -e -x

# ── Environment ───────────────────────────────────────────────────
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_PROCID}
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_MODE="${WANDB_MODE}"

# ── Rendezvous ────────────────────────────────────────────────────
export MASTER_PORT=${MASTER_PORT}
nodes=( \$(scontrol show hostnames \$SLURM_JOB_NODELIST) )
export MASTER_ADDR=\${nodes[0]}
export NODE_RANK=\$SLURM_PROCID

echo "MASTER_ADDR: \$MASTER_ADDR"
echo "NODE_RANK:   \$NODE_RANK"

# ── Launch ────────────────────────────────────────────────────────
srun torchrun \\
    --nnodes \$SLURM_JOB_NUM_NODES \\
    --nproc_per_node ${NUM_GPUS} \\
    --node_rank \$SLURM_PROCID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\$MASTER_ADDR:\$MASTER_PORT" \\
    fastvideo/train/entrypoint/train.py \\
    --config ${CONFIG} \\
    --training.distributed.num_gpus ${TOTAL_GPUS} \\
    ${EXTRA_ARGS[*]:-}
EOF
