#!/bin/bash
#SBATCH --job-name=matrixgame2_sf_dmd
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=logs/slurm/matrixgame2_sf_dmd_%j.out
#SBATCH --error=logs/slurm/matrixgame2_sf_dmd_%j.err
#SBATCH --exclusive

set -euo pipefail
set -x

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_BIN="${CONDA_ROOT}/bin/conda"
ENV_NAME="${ENV_NAME:-FastVideo_kaiqin}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
TOTAL_GPUS="$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))"

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_PROCID}"
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY
export WANDB_MODE

if [ -x "${CONDA_BIN}" ]; then
    eval "$("${CONDA_BIN}" shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "conda not found at ${CONDA_BIN}" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT
export NODE_RANK="${SLURM_PROCID}"

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "TOTAL_GPUS: ${TOTAL_GPUS}"

RUN_NAME="$(date +"%m%d_%H%M")"
echo "RUN_NAME: ${RUN_NAME}"

GENERATOR_MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers"
REAL_SCORE_MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"
FAKE_SCORE_MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"

DATA_DIR="data/matrixgame2"
VALIDATION_DATASET_FILE="examples/distill/MatrixGame2.0/validation.json"

srun python -m torch.distributed.run \
  --nnodes "${SLURM_JOB_NUM_NODES}" \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank "${SLURM_PROCID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  fastvideo/training/matrixgame2_self_forcing_distillation_pipeline.py \
  --config examples/distill/MatrixGame2.0/distill_dmd_slurm.yaml \
  --model_path "${GENERATOR_MODEL_PATH}" \
  --engine.num_gpus "${TOTAL_GPUS}" \
  --engine.parallelism.hsdp_shard_dim "${TOTAL_GPUS}" \
  --training.distillation.real_score_model_path "${REAL_SCORE_MODEL_PATH}" \
  --training.distillation.fake_score_model_path "${FAKE_SCORE_MODEL_PATH}" \
  --training.data.data_path "${DATA_DIR}" \
  --training.checkpoint.output_dir "checkpoints/matrixgame2_sf_${RUN_NAME}" \
  --training.tracker.run_name "${RUN_NAME}_test" \
  --training.validation.dataset_file "${VALIDATION_DATASET_FILE}"
