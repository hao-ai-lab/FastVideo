#!/usr/bin/env bash
set -euo pipefail
: "${RUN_ROOT:?}" "${STAGE:?}" "${MASTER_PORT:?}" "${WANDB_API_KEY:?}"
REPO="${REPO:-/mnt/nfs/vlm-k1kong/FastVideo-openvid-a12-a15-final-20260717}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
REQUIRED_ANCESTOR="30fcfae6578e2f5776d86cafe807f4a58722e26e"
STAGE_DIR="$RUN_ROOT/$STAGE"
CONFIG="$STAGE_DIR/config/run.yaml"
STATE="$STAGE_DIR/state"
LOG="$STAGE_DIR/logs/train.log"
git -C "$REPO" merge-base --is-ancestor "$REQUIRED_ANCESTOR" HEAD
mkdir -p "$STAGE_DIR"/{logs,state,checkpoints,validation,tracker}

export HF_HOME=/mnt/lustre/vlm-k1kong/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
export XDG_CACHE_HOME=/mnt/lustre/vlm-k1kong/xdg-cache
export CUDA_CACHE_PATH=/mnt/lustre/vlm-k1kong/cuda-cache
export TRITON_CACHE_DIR="/mnt/lustre/vlm-k1kong/triton-cache/openvid-a12-a15/${HOSTNAME}/${CONDITION}/${STAGE}"
export TORCHINDUCTOR_CACHE_DIR="/mnt/lustre/vlm-k1kong/torchinductor-cache/openvid-a12-a15/${HOSTNAME}/${CONDITION}/${STAGE}"
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_MODE=online WANDB_ENTITY=kaiqin_kong_ucsd
export WANDB_PROJECT=causal_forcing_openvid_a12_a15 WANDB_RESUME=allow
export TOKENIZERS_PARALLELISM=false FASTVIDEO_DIST_TIMEOUT_MINUTES=120
export TORCH_NCCL_BLOCKING_WAIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 VIRTUAL_ENV="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH" PYTHONPATH="$REPO"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

if [[ -s "$STATE/wandb_run_id" ]]; then
  export WANDB_RUN_ID="$(<"$STATE/wandb_run_id")"
else
  export WANDB_RUN_ID="$($ENV_DIR/bin/python -c 'import wandb; print(wandb.util.generate_id())')"
  printf '%s\n' "$WANDB_RUN_ID" > "$STATE/wandb_run_id"
fi
resume=()
if find "$STAGE_DIR/checkpoints" -mindepth 2 -maxdepth 2 -type d -name dcp -print -quit | grep -q .; then
  resume+=(--training.checkpoint.resume_from_checkpoint latest)
fi
cmd=("$ENV_DIR/bin/python" -m torch.distributed.run --nnodes 1 --node_rank 0
  --nproc_per_node 4 --master_addr 127.0.0.1 --master_port "$MASTER_PORT"
  -m fastvideo.train.entrypoint.train --config "$CONFIG" "${resume[@]}")
printf 'running\n' > "$STATE/status"; date -Is > "$STATE/started_at"
cd "$REPO"; set +e; "${cmd[@]}" 2>&1 | tee -a "$LOG"; rc=${PIPESTATUS[0]}; set -e
printf '%s\n' "$rc" > "$STATE/exit_code"; date -Is > "$STATE/finished_at"
if [[ "$rc" -eq 0 ]]; then printf 'completed\n' > "$STATE/status"; else printf 'failed\n' > "$STATE/status"; fi
exit "$rc"
