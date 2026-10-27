#!/usr/bin/env bash
# Submit a multi-node Slurm training job.
#
# Usage:
#   bash examples/train/run_slurm.sh <config.yaml> <num_nodes> [--dotted.key value ...]
#
# Examples:
#   bash examples/train/run_slurm.sh examples/train/configs/example.yaml 2
#   bash examples/train/run_slurm.sh examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml 4 \
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
#   WANDB_API_KEY   Exported W&B API key required for online logging
#   WANDB_MODE      W&B mode                  (default: online)
#   SBATCH_CONSTRAINT Slurm node constraint    (default: "")
#   SBATCH_TIMELIMIT Slurm time limit          (default: "")

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
WANDB_MODE="${WANDB_MODE:-online}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"
SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-}"

TOTAL_GPUS=$(( NUM_NODES * NUM_GPUS ))

CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
JOB_NAME="${JOB_NAME:-${CONFIG_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/slurm}"
mkdir -p "${OUTPUT_DIR}"

# ── Build sbatch args ─────────────────────────────────────────────
# Forward the exported W&B API key through Slurm's environment so the generated
# job script and its xtrace output do not contain the credential value.
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
    --export="ALL,WANDB_MODE=${WANDB_MODE}"
    --exclusive
)

if [[ -n "${EXCLUDE}" ]]; then
    SBATCH_ARGS+=(--exclude="${EXCLUDE}")
fi
# Apply hardware and wall-clock requirements selected by an experiment launcher.
if [[ -n "${SBATCH_CONSTRAINT}" ]]; then
    SBATCH_ARGS+=(--constraint="${SBATCH_CONSTRAINT}")
fi
if [[ -n "${SBATCH_TIMELIMIT}" ]]; then
    SBATCH_ARGS+=(--time="${SBATCH_TIMELIMIT}")
fi

# ── Collect extra overrides for the training script ───────────────
EXTRA_ARGS=("$@")
# Shell-escape values before embedding them in the generated Bash script so
# config paths and override arguments retain their original boundaries.
printf -v CONFIG_SHELL_ARG '%q' "${CONFIG}"
EXTRA_ARGS_SHELL=""
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    printf -v EXTRA_ARGS_SHELL ' %q' "${EXTRA_ARGS[@]}"
fi

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
export TOKENIZERS_PARALLELISM=false

# ── Rendezvous ────────────────────────────────────────────────────
export MASTER_PORT=${MASTER_PORT}
nodes=( \$(scontrol show hostnames \$SLURM_JOB_NODELIST) )
export MASTER_ADDR=\${nodes[0]}
# ── Launch ────────────────────────────────────────────────────────
# Resolve SLURM_PROCID inside each srun task so every node receives a distinct
# torchrun node rank and Triton cache directory.
launch_training_node() {
    export TRITON_CACHE_DIR="/tmp/triton_cache_\${SLURM_PROCID}"
    echo "MASTER_ADDR: \${MASTER_ADDR}"
    echo "NODE_RANK:   \${SLURM_PROCID}"
    exec torchrun \\
        --nnodes "\${SLURM_JOB_NUM_NODES}" \\
        --nproc_per_node ${NUM_GPUS} \\
        --node_rank "\${SLURM_PROCID}" \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
        fastvideo/train/entrypoint/train.py \\
        --config ${CONFIG_SHELL_ARG} \\
        --training.distributed.num_gpus ${TOTAL_GPUS}${EXTRA_ARGS_SHELL}
}
export -f launch_training_node
srun bash -c launch_training_node
EOF
