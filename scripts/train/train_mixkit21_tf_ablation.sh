#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/nfs/vlm-k1kong/FastVideo-causal-kernel}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-examples/train/configs/ablation/wan_causal_mixkit21/tf_2k_template.yaml}"
MATRIX="${MATRIX:-examples/train/configs/ablation/wan_causal_mixkit21/experiment_matrix.tsv}"
CONDITION="${CONDITION:?Set CONDITION to an experiment ID such as A01}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the persistent condition root}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29800}"
PROJECT_NAME="${PROJECT_NAME:-causal_forcing_mixkit21_kernel_tf2k_long249_ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-kaiqin_kong_ucsd}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-tf2k_${CONDITION}_$(basename "$RUN_ROOT")}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000}"
VALIDATION_EVERY_STEPS="${VALIDATION_EVERY_STEPS:-200}"
VALIDATION_SAMPLING_STEPS="${VALIDATION_SAMPLING_STEPS:-40}"
VALIDATION_DATASET_FILE="${VALIDATION_DATASET_FILE:-examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Python environment is missing: $ENV_DIR" >&2
  exit 2
fi
if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY at runtime." >&2
  exit 2
fi

STAGE_DIR="$RUN_ROOT/tf"
CONFIG_DIR="$STAGE_DIR/config"
SCRIPT_DIR="$STAGE_DIR/scripts"
LOG_DIR="$STAGE_DIR/logs"
CHECKPOINT_DIR="$STAGE_DIR/checkpoints"
VALIDATION_DIR="$STAGE_DIR/validation"
STATE_DIR="$STAGE_DIR/state"
RUN_CONFIG="$CONFIG_DIR/run.yaml"
LOG_FILE="$LOG_DIR/train.log"

mkdir -p "$CONFIG_DIR" "$SCRIPT_DIR" "$LOG_DIR" "$CHECKPOINT_DIR" \
  "$VALIDATION_DIR" "$STATE_DIR"
ln -sfn "$CHECKPOINT_DIR/tracker" "$STAGE_DIR/tracker"

cp "$REPO/$CONFIG_TEMPLATE" "$CONFIG_DIR/source_template.yaml"
cp "$REPO/$MATRIX" "$CONFIG_DIR/experiment_matrix.tsv"
cp "$REPO/scripts/train/manage_mixkit21_tf_ablation.py" "$SCRIPT_DIR/"
cp "$REPO/scripts/train/train_mixkit21_tf_ablation.sh" "$SCRIPT_DIR/"

"$ENV_DIR/bin/python" "$REPO/scripts/train/manage_mixkit21_tf_ablation.py" \
  --template "$REPO/$CONFIG_TEMPLATE" \
  --matrix "$REPO/$MATRIX" \
  render \
  --condition "$CONDITION" \
  --output "$RUN_CONFIG" \
  --run-root "$RUN_ROOT" \
  --project-name "$PROJECT_NAME" \
  --run-name "$WANDB_RUN_NAME" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --checkpoint-steps "$CHECKPOINT_STEPS" \
  --validation-every-steps "$VALIDATION_EVERY_STEPS" \
  --validation-sampling-steps "$VALIDATION_SAMPLING_STEPS" \
  --validation-dataset-file "$REPO/$VALIDATION_DATASET_FILE"

if [[ -s "$STATE_DIR/wandb_run_id" ]]; then
  WANDB_RUN_ID="$(<"$STATE_DIR/wandb_run_id")"
else
  WANDB_RUN_ID="$("$ENV_DIR/bin/python" -c 'import wandb; print(wandb.util.generate_id())')"
  printf '%s\n' "$WANDB_RUN_ID" > "$STATE_DIR/wandb_run_id"
fi

export HF_HOME="${HF_HOME:-/mnt/lustre/vlm-k1kong/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-$HF_HOME/diffusers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/lustre/vlm-k1kong/xdg-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/mnt/lustre/vlm-k1kong/cuda-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/lustre/vlm-k1kong/triton-cache/mixkit21-kernel/${HOSTNAME:-unknown}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/lustre/vlm-k1kong/torchinductor-cache/mixkit21-kernel/${HOSTNAME:-unknown}}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY PROJECT_NAME WANDB_RUN_ID
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export FASTVIDEO_DIST_TIMEOUT_MINUTES="${FASTVIDEO_DIST_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export VIRTUAL_ENV="$ENV_DIR"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$DIFFUSERS_CACHE" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

WANDB_URL="https://wandb.ai/$WANDB_ENTITY/$PROJECT_NAME/runs/$WANDB_RUN_ID"
cat > "$STAGE_DIR/manifest.env" <<EOF
condition=$CONDITION
repo=$REPO
run_root=$RUN_ROOT
run_config=$RUN_CONFIG
log_file=$LOG_FILE
num_gpus=$NUM_GPUS
master_port=$MASTER_PORT
max_train_steps=$MAX_TRAIN_STEPS
validation_frames=249
training_latent_frames=21
validation_latent_frames=63
wandb_project=$PROJECT_NAME
wandb_entity=$WANDB_ENTITY
wandb_run_id=$WANDB_RUN_ID
wandb_url=$WANDB_URL
wandb_api_key_set=$([[ -n "${WANDB_API_KEY:-}" ]] && echo true || echo false)
EOF

resume_args=()
if find "$CHECKPOINT_DIR" -mindepth 2 -maxdepth 2 -type d -name dcp -print -quit | grep -q .; then
  resume_args+=(--training.checkpoint.resume_from_checkpoint latest)
fi

cmd=(
  "$ENV_DIR/bin/python" -m torch.distributed.run
  --nnodes 1
  --node_rank 0
  --nproc_per_node "$NUM_GPUS"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  -m fastvideo.train.entrypoint.train
  --config "$RUN_CONFIG"
  "${resume_args[@]}"
)

printf 'running\n' > "$STATE_DIR/status"
date -Is > "$STATE_DIR/started_at"
echo "condition=$CONDITION run_root=$RUN_ROOT wandb=$WANDB_URL"
printf 'command:'
printf ' %q' "${cmd[@]}"
echo

cd "$REPO"
set +e
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

printf '%s\n' "$rc" > "$STATE_DIR/exit_code"
date -Is > "$STATE_DIR/finished_at"
if [[ "$rc" -eq 0 ]]; then
  printf 'completed\n' > "$STATE_DIR/status"
else
  printf 'failed\n' > "$STATE_DIR/status"
fi
exit "$rc"
