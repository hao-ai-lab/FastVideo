#!/bin/bash
# H3 joint video+audio TDM with VSA on one four-GPU GB200 node via Slurm.
#
# Runs the tdm-port branch's H3 TDM config with the video-sparse backend on the
# student. Because TDM prepares `latents_source="zeros"`, the dataset's latent
# *values* are unused: the trajectory geometry comes from the training config,
# so the 480x832 canvas (the drifting-validated geometry, ~2.6x fewer video
# tokens than 768x1344) is a plain config override and needs no new asset.
#
# Slurm invocation:
#   sbatch -p all --gres=gpu:nvidia_gb200:4 \
#       --export=ALL,MODE=smoke,MAX_STEPS=2 ...
#   MODE=smoke (2 steps, no validation) or MODE=gate (config steps + report).
#
# The dev image is provided through Pyxis/enroot; /mnt/lustre is bind-mounted at
# /workspace so every path in the shipped config resolves exactly as it did in
# the Kubernetes runs.
#SBATCH --job-name=h3-tdm-vsa
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia_gb200:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/mnt/lustre/vlm-mal004/tdm-port/slurm-%x-%j.log

set -Eeuo pipefail

# Positional parameters (passed as `sbatch --export=NONE script MODE ...`).
# Slurm environment export fails for this account
# (`user_env_retrieval_failed_requeued_held`), so the job takes everything it
# needs as arguments and sets its own environment.
MODE=${1:-smoke}
MAX_STEPS=${2:-2}
SPARSITY=${3:-0.5}
HEIGHT=${4:-480}
WIDTH=${5:-832}
RUN_NAME=${6:-h3-tdm-vsa-${MODE}}
EXTRA_ARGS=${7:-}

IMAGE=${IMAGE:-ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest}
CODE=${CODE:-/workspace/vlm-mal004/tdm-port/code}
RUN_ROOT=/workspace/run/tdm-port/${RUN_NAME}

echo "START ${RUN_NAME} $(date -u +%FT%TZ) mode=${MODE} geom=${HEIGHT}x${WIDTH} sparsity=${SPARSITY}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun --container-image="${IMAGE}" \
     --container-mounts=/mnt/lustre:/workspace \
     --container-workdir="${CODE}" \
     bash -lc "
set -Eeuo pipefail
export HOME=/tmp/tdm-home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/issue-775/shared-cache/hf
export HF_HUB_CACHE=\$HF_HOME/hub
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
export TORCHDYNAMO_DISABLE=1
export TRITON_CACHE_DIR=/tmp/triton-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${CODE}
mkdir -p \"\$HOME\" \"\$TRITON_CACHE_DIR\" \"${RUN_ROOT}/logs\"
cd ${CODE}
mkdir -p ${RUN_ROOT}
/opt/venv/bin/torchrun --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml \
    --models.student.attention_backend VIDEO_SPARSE_ATTN_H3 \
    --vsa.sparsity ${SPARSITY} \
    --training.data.num_height ${HEIGHT} \
    --training.data.num_width ${WIDTH} \
    --training.loop.max_train_steps ${MAX_STEPS} \
    ${EXTRA_ARGS:-} \
    --training.checkpoint.output_dir ${RUN_ROOT}/output > ${RUN_ROOT}/logs/train.log 2>&1

grep -q 'Training completed' ${RUN_ROOT}/logs/train.log
tail -2 ${RUN_ROOT}/logs/train.log
printf 'JOB_OK %s\n' \"\$(date -u +%FT%TZ)\"
"
echo "DONE ${RUN_NAME} $(date -u +%FT%TZ)"
