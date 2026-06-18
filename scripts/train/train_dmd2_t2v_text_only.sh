#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml}"
DATA_PATH="${DATA_PATH:-data/train_text_only_dmd_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wan2.1_dmd2_text_only}"
NUM_GPUS="${NUM_GPUS:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29521}"
SP_SIZE="${SP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
HSDP_REPLICATE_DIM="${HSDP_REPLICATE_DIM:-1}"
HSDP_SHARD_DIM="${HSDP_SHARD_DIM:-$NUM_GPUS}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_FRAMES="${NUM_FRAMES:-1}"
NUM_LATENT_T="${NUM_LATENT_T:-1}"
VALIDATION_NUM_FRAMES="${VALIDATION_NUM_FRAMES:-$NUM_FRAMES}"
VALIDATION_PROMPT_FILE="${VALIDATION_PROMPT_FILE:-}"
VALIDATION_FILE="${VALIDATION_FILE:-examples/train/configs/distribution_matching/wan/dmd2_text_only_validation.json}"
VALIDATION_OFFLOAD_TRAINING_STATE="${VALIDATION_OFFLOAD_TRAINING_STATE:-true}"
VALIDATION_UNLOAD_PIPELINE_AFTER="${VALIDATION_UNLOAD_PIPELINE_AFTER:-true}"
CFG_UNCOND_TEXT="${CFG_UNCOND_TEXT:-zero}"
CFG_UNCOND_ON_MISSING="${CFG_UNCOND_ON_MISSING:-ignore}"
PROJECT_NAME="${PROJECT_NAME:-distillation_wan_text_only}"
RUN_NAME="${RUN_NAME:-wan2.1_dmd2_text_only}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fastvideo}"
LOG_DIR="${LOG_DIR:-logs/train}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Training config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
    echo "Preprocessed dataset directory not found: $DATA_PATH" >&2
    echo "Run scripts/preprocess/preprocess_train_text_only_dmd.sh first." >&2
    exit 1
fi

num_parquet=$(find "$DATA_PATH" -name '*.parquet' | wc -l | tr -d ' ')
if [[ "$num_parquet" -eq 0 ]]; then
    echo "No parquet files found under $DATA_PATH" >&2
    echo "Run scripts/preprocess/preprocess_train_text_only_dmd.sh first." >&2
    exit 1
fi

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda activation script not found under $CONDA_ROOT" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
    if ! python -c "import hf_transfer" >/dev/null 2>&1; then
        echo "HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed; disabling fast transfer."
        export HF_HUB_ENABLE_HF_TRANSFER=0
    fi
else
    export HF_HUB_ENABLE_HF_TRANSFER=0
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_dmd2_text_only}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ -n "$VALIDATION_PROMPT_FILE" ]]; then
    if [[ ! -f "$VALIDATION_PROMPT_FILE" ]]; then
        echo "Validation prompt file not found: $VALIDATION_PROMPT_FILE" >&2
        echo "Set VALIDATION_PROMPT_FILE to a text file, or leave it empty and set VALIDATION_FILE=<validation.json>." >&2
        exit 1
    fi
    python - "$VALIDATION_PROMPT_FILE" "$VALIDATION_FILE" <<'PY'
import json
import os
import sys

prompt_file, validation_file = sys.argv[1:3]
with open(prompt_file, encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

if not prompts:
    raise SystemExit(f"No validation prompts found in {prompt_file}")

validation_dir = os.path.dirname(os.path.abspath(validation_file))
if validation_dir:
    os.makedirs(validation_dir, exist_ok=True)

with open(validation_file, "w", encoding="utf-8") as f:
    json.dump(
        {"data": [{"caption": prompt} for prompt in prompts]},
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")

print(f"Wrote {len(prompts)} validation prompts to {validation_file}")
PY
elif [[ ! -f "$VALIDATION_FILE" ]]; then
    echo "Validation dataset file not found: $VALIDATION_FILE" >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/dmd2_t2v_text_only_${timestamp}.log"

cmd=(
    torchrun
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --nproc_per_node "$NUM_GPUS"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    -m fastvideo.train.entrypoint.train
    --config "$CONFIG"
    --training.data.data_path "$DATA_PATH"
    --training.data.preprocessed_data_type text_only
    --training.data.dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --training.data.num_frames "$NUM_FRAMES"
    --training.data.num_latent_t "$NUM_LATENT_T"
    --callbacks.validation.dataset_file "$VALIDATION_FILE"
    --callbacks.validation.num_frames "$VALIDATION_NUM_FRAMES"
    --callbacks.validation.offload_training_state "$VALIDATION_OFFLOAD_TRAINING_STATE"
    --callbacks.validation.unload_pipeline_after_validation "$VALIDATION_UNLOAD_PIPELINE_AFTER"
    --method.cfg_uncond.text "$CFG_UNCOND_TEXT"
    --method.cfg_uncond.on_missing "$CFG_UNCOND_ON_MISSING"
    --training.distributed.num_gpus "$NUM_GPUS"
    --training.distributed.sp_size "$SP_SIZE"
    --training.distributed.tp_size "$TP_SIZE"
    --training.distributed.hsdp_replicate_dim "$HSDP_REPLICATE_DIM"
    --training.distributed.hsdp_shard_dim "$HSDP_SHARD_DIM"
    --training.checkpoint.output_dir "$OUTPUT_DIR"
    --training.tracker.project_name "$PROJECT_NAME"
    --training.tracker.run_name "$RUN_NAME"
)

echo "DMD2 T2V text-only training config:"
echo "  config: $CONFIG"
echo "  data path: $DATA_PATH"
echo "  parquet files: $num_parquet"
echo "  output dir: $OUTPUT_DIR"
echo "  frames / latent T: $NUM_FRAMES / $NUM_LATENT_T"
echo "  validation prompt file: ${VALIDATION_PROMPT_FILE:-<none>}"
echo "  validation dataset: $VALIDATION_FILE"
echo "  validation frames: $VALIDATION_NUM_FRAMES"
echo "  validation offload training state: $VALIDATION_OFFLOAD_TRAINING_STATE"
echo "  validation unload pipeline after: $VALIDATION_UNLOAD_PIPELINE_AFTER"
echo "  GPUs: $NUM_GPUS"
echo "  SP/TP: $SP_SIZE/$TP_SIZE"
echo "  HSDP replicate/shard: $HSDP_REPLICATE_DIM/$HSDP_SHARD_DIM"
echo "  CFG uncond text/on_missing: $CFG_UNCOND_TEXT/$CFG_UNCOND_ON_MISSING"
echo "  W&B mode: $WANDB_MODE"
echo "  log file: $log_file"
echo "Command:"
printf '  %q' "${cmd[@]}" "$@"
echo

"${cmd[@]}" "$@" 2>&1 | tee "$log_file"
