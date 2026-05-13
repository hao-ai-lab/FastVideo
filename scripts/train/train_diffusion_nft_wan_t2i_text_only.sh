#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -x "$REPO_ROOT/.venv/bin/torchrun" ]]; then
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$REPO_ROOT/DiffusionNFT" ]]; then
    export PYTHONPATH="$REPO_ROOT/DiffusionNFT:$PYTHONPATH"
fi
TORCHRUN="${TORCHRUN:-torchrun}"

CONFIG="${CONFIG:-examples/train/configs/diffusion_nft/wan/t2i_text_only.yaml}"
PROMPT_FILE="${PROMPT_FILE:-DiffusionNFT/dataset/pickscore/train.txt}"
DATA_PATH="${DATA_PATH:-data/train_text_only_diffusion_nft_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wan2.1_diffusion_nft_text_only}"
AUTO_PREPROCESS="${AUTO_PREPROCESS:-true}"
PREPROCESS_GPU_NUM="${PREPROCESS_GPU_NUM:-1}"
PREPROCESS_BATCH_SIZE="${PREPROCESS_BATCH_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
SP_SIZE="${SP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
HSDP_REPLICATE_DIM="${HSDP_REPLICATE_DIM:-1}"
HSDP_SHARD_DIM="${HSDP_SHARD_DIM:-$NUM_GPUS}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_HEIGHT="${NUM_HEIGHT:-448}"
NUM_WIDTH="${NUM_WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-1}"
NUM_LATENT_T="${NUM_LATENT_T:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-2}"
INNER_EPOCHS="${INNER_EPOCHS:-1}"
SAMPLE_TIMESTEPS="${SAMPLE_TIMESTEPS:-[1000,750,500,250]}"
SAMPLE_GUIDANCE_SCALE="${SAMPLE_GUIDANCE_SCALE:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-6}"
PROJECT_NAME="${PROJECT_NAME:-diffusion_nft_wan_text_only}"
RUN_NAME="${RUN_NAME:-wan2.1_diffusion_nft_text_only}"
LOG_DIR="${LOG_DIR:-logs/train}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Training config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

num_parquet=0
if [[ -d "$DATA_PATH" ]]; then
    num_parquet=$(find "$DATA_PATH" -name '*.parquet' | wc -l | tr -d ' ')
fi

if [[ "$num_parquet" -eq 0 ]]; then
    if [[ "$AUTO_PREPROCESS" != "true" ]]; then
        echo "No parquet files found under $DATA_PATH" >&2
        echo "Set AUTO_PREPROCESS=true or run scripts/preprocess/preprocess_train_text_only_dmd.sh first." >&2
        exit 1
    fi
    echo "No text-only parquet found under $DATA_PATH; running text-only preprocessing."
    OUTPUT_DIR="$DATA_PATH" \
        GPU_NUM="$PREPROCESS_GPU_NUM" \
        BATCH_SIZE="$PREPROCESS_BATCH_SIZE" \
        bash scripts/preprocess/preprocess_train_text_only_dmd.sh "$PROMPT_FILE"
    num_parquet=$(find "$DATA_PATH" -name '*.parquet' | wc -l | tr -d ' ')
fi

if [[ "$num_parquet" -eq 0 ]]; then
    echo "No parquet files were produced under $DATA_PATH" >&2
    exit 1
fi

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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_diffusion_nft}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/diffusion_nft_wan_t2i_text_only_${timestamp}.log"

cmd=(
    "$TORCHRUN"
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
    --training.data.num_height "$NUM_HEIGHT"
    --training.data.num_width "$NUM_WIDTH"
    --training.data.num_frames "$NUM_FRAMES"
    --training.data.num_latent_t "$NUM_LATENT_T"
    --training.distributed.num_gpus "$NUM_GPUS"
    --training.distributed.sp_size "$SP_SIZE"
    --training.distributed.tp_size "$TP_SIZE"
    --training.distributed.hsdp_replicate_dim "$HSDP_REPLICATE_DIM"
    --training.distributed.hsdp_shard_dim "$HSDP_SHARD_DIM"
    --training.optimizer.learning_rate "$LEARNING_RATE"
    --training.loop.max_train_steps "$MAX_TRAIN_STEPS"
    --training.checkpoint.output_dir "$OUTPUT_DIR"
    --training.tracker.project_name "$PROJECT_NAME"
    --training.tracker.run_name "$RUN_NAME"
    --method.train_batch_size "$TRAIN_BATCH_SIZE"
    --method.num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
    --method.inner_epochs "$INNER_EPOCHS"
    --method.sample_timesteps "$SAMPLE_TIMESTEPS"
    --method.sample_guidance_scale "$SAMPLE_GUIDANCE_SCALE"
)

echo "DiffusionNFT Wan text-to-image training config:"
echo "  config: $CONFIG"
echo "  prompt file: $PROMPT_FILE"
echo "  data path: $DATA_PATH"
echo "  parquet files: $num_parquet"
echo "  output dir: $OUTPUT_DIR"
echo "  image size: ${NUM_HEIGHT}x${NUM_WIDTH}, latent T: $NUM_LATENT_T"
echo "  max train steps: $MAX_TRAIN_STEPS"
echo "  sample timesteps: $SAMPLE_TIMESTEPS"
echo "  images per prompt: $NUM_IMAGES_PER_PROMPT"
echo "  GPUs: $NUM_GPUS"
echo "  SP/TP: $SP_SIZE/$TP_SIZE"
echo "  W&B mode: $WANDB_MODE"
echo "  log file: $log_file"
echo "Command:"
printf '  %q' "${cmd[@]}" "$@"
echo

"${cmd[@]}" "$@" 2>&1 | tee "$log_file"
