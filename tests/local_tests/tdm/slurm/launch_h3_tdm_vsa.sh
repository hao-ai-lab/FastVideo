#!/bin/bash
# Launch the H3 TDM VSA job on Slurm from the login node.
#
# `sbatch` submission fails site-side with
# `user_env_retrieval_failed_requeued_held` (the compute nodes run a minimal OS
# and the retrieval helper cannot run there), while a detached `srun` step
# works. This launcher therefore runs the payload through `srun` under
# setsid/nohup and logs to Lustre.
#
# Usage: launch_h3_tdm_vsa.sh MODE MAX_STEPS SPARSITY HEIGHT WIDTH RUN_NAME [EXTRA_ARGS]
set -Eeuo pipefail

MODE=${1:-smoke}
MAX_STEPS=${2:-2}
SPARSITY=${3:-0.5}
HEIGHT=${4:-480}
WIDTH=${5:-832}
RUN_NAME=${6:-h3-tdm-vsa-${MODE}}
EXTRA_ARGS=${7:-}

IMAGE=ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest
CODE=/workspace/vlm-mal004/tdm-port/code
LOG=/mnt/lustre/vlm-mal004/tdm-port/srun-${RUN_NAME}.log

cd /mnt/lustre/vlm-mal004/tdm-port/code
setsid nohup srun --export=NONE -p all --gres=gpu:nvidia_gb200:4 -N1 -n1 \
    --container-image="${IMAGE}" \
    --container-mounts=/mnt/lustre:/workspace \
    --container-workdir="${CODE}" \
    bash -lc "${CODE}/tests/local_tests/tdm/slurm/h3_tdm_vsa_run.sh ${MODE} ${MAX_STEPS} ${SPARSITY} ${HEIGHT} ${WIDTH} ${RUN_NAME} '${EXTRA_ARGS}'" \
    > "${LOG}" 2>&1 < /dev/null &
echo "launched ${RUN_NAME} (log: ${LOG})"
