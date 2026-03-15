#!/bin/bash
#SBATCH --job-name=wg-sf
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=examples/train/slurm_%j.out
#SBATCH --error=examples/train/slurm_%j.err
#SBATCH --exclusive

set -euo pipefail

CONFIG="${1:?Usage: sbatch examples/train/run.slurm <config.yaml> [extra flags...]}"
shift
EXTRA_ARGS=("$@")
set --

cd /mnt/weka/home/hao.zhang/mhuo/FastVideo-refactor

get_num_gpus() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        echo "${NUM_GPUS}"
        return
    fi
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        echo "${SLURM_GPUS_ON_NODE%%(*}"
        return
    fi
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
        echo "${SLURM_GPUS_PER_NODE%%(*}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d " "
    else
        echo 8
    fi
}

export NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
export NUM_GPUS="$(get_num_gpus)"
export MASTER_PORT="${MASTER_PORT:-29501}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
    nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
    export MASTER_ADDR="${nodes[0]}"
fi

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="${WANDB_API_KEY:-7ff8b6e8356924f7a6dd51a0342dd1a422ea9352}"
export WANDB_MODE="${WANDB_MODE:-online}"

set +u
source ~/conda/miniconda/bin/activate
conda activate mhuo-fv
set -u
export PYTHONPATH="/mnt/weka/home/hao.zhang/mhuo/FastVideo-refactor:${PYTHONPATH:-}"

echo "=== Distillation Training (Slurm) ==="
echo "Config:      ${CONFIG}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Num Nodes:   ${NNODES}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Extra args:  ${EXTRA_ARGS[*]:-}"
echo "====================================="

srun torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NUM_GPUS}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --node_rank "${SLURM_PROCID}" \
    fastvideo/train/entrypoint/train.py \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
