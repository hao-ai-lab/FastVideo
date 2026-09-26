#!/bin/bash
# H3 TDM VSA payload: runs inside the FastVideo dev container on a Slurm node.
#
# Arguments: MODE MAX_STEPS SPARSITY HEIGHT WIDTH RUN_NAME [BACKEND] [EXTRA_ARGS]
# MODE=smoke runs a couple of steps with validation off; MODE=gate runs the
# config's full step count, so the final-step validation fires and the run
# yields acceptance samples. BACKEND defaults to the video-sparse backend;
# pass TORCH_SDPA for the dense control at the same geometry.
set -Eeuo pipefail

MODE=${1:-smoke}
MAX_STEPS=${2:-2}
SPARSITY=${3:-0.5}
HEIGHT=${4:-480}
WIDTH=${5:-832}
RUN_NAME=${6:-h3-tdm-vsa-${MODE}}
BACKEND=${7:-VIDEO_SPARSE_ATTN_H3}
EXTRA_ARGS=${8:-}

CODE=/workspace/vlm-mal004/tdm-port/code
# The shared /workspace/run tree was created by the Kubernetes jobs as root and
# is not writable from Slurm, so run outputs live under this account's Lustre
# directory.
RUN_ROOT=/workspace/vlm-mal004/tdm-port/runs/${RUN_NAME}

export HOME=/tmp/tdm-home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/issue-775/shared-cache/hf
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
export TORCHDYNAMO_DISABLE=1
export TRITON_CACHE_DIR=/tmp/triton-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${CODE}"
mkdir -p "${HOME}" "${TRITON_CACHE_DIR}" "${RUN_ROOT}/logs"

cd "${CODE}"
test -d data/tdm_t2v_overfit_text_only || true

printf 'PAYLOAD_START %s mode=%s steps=%s geom=%sx%s sparsity=%s\n' \
    "$(date -u +%FT%TZ)" "${MODE}" "${MAX_STEPS}" "${HEIGHT}" "${WIDTH}" "${SPARSITY}"

/opt/venv/bin/torchrun --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml \
    --models.student.attention_backend "${BACKEND}" \
    --vsa.sparsity "${SPARSITY}" \
    --training.data.num_height "${HEIGHT}" \
    --training.data.num_width "${WIDTH}" \
    --training.loop.max_train_steps "${MAX_STEPS}" \
    ${EXTRA_ARGS} \
    --training.checkpoint.output_dir "${RUN_ROOT}/output" > "${RUN_ROOT}/logs/train.log" 2>&1

grep -q 'Training completed' "${RUN_ROOT}/logs/train.log"
tail -2 "${RUN_ROOT}/logs/train.log"
printf 'PAYLOAD_OK %s\n' "$(date -u +%FT%TZ)"
