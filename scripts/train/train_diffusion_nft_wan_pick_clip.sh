#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-examples/train/configs/rl/wan/diffusion_nft_pick_clip.yaml}"
DATA_PATH="${DATA_PATH:-data/pickscore_text_only_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wan2.1_diffusion_nft_pick_clip}"
NUM_GPUS="${NUM_GPUS:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
SP_SIZE="${SP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
HSDP_REPLICATE_DIM="${HSDP_REPLICATE_DIM:-1}"
HSDP_SHARD_DIM="${HSDP_SHARD_DIM:-$NUM_GPUS}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_FRAMES="${NUM_FRAMES:-1}"
NUM_LATENT_T="${NUM_LATENT_T:-1}"
PROJECT_NAME="${PROJECT_NAME:-diffusion_nft_wan}"
RUN_NAME="${RUN_NAME:-wan2.1_diffusion_nft_pick_clip}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fastvideo}"
LOG_DIR="${LOG_DIR:-logs/train}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Training config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
    echo "Preprocessed dataset directory not found: $DATA_PATH" >&2
    echo "Run preprocessing first, for example:" >&2
    echo "  GPU_NUM=4 BATCH_SIZE=1 OUTPUT_DIR=$DATA_PATH \\" >&2
    echo "    bash scripts/preprocess/preprocess_train_text_only_dmd.sh DiffusionNFT/dataset/pickscore/train.txt" >&2
    exit 1
fi

num_parquet=$(find "$DATA_PATH" -name '*.parquet' | wc -l | tr -d ' ')
if [[ "$num_parquet" -eq 0 ]]; then
    echo "No parquet files found under $DATA_PATH" >&2
    echo "Run preprocessing first, for example:" >&2
    echo "  GPU_NUM=4 BATCH_SIZE=1 OUTPUT_DIR=$DATA_PATH \\" >&2
    echo "    bash scripts/preprocess/preprocess_train_text_only_dmd.sh DiffusionNFT/dataset/pickscore/train.txt" >&2
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
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_diffusion_nft_wan}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/diffusion_nft_wan_pick_clip_${timestamp}.log"

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
    --training.distributed.num_gpus "$NUM_GPUS"
    --training.distributed.sp_size "$SP_SIZE"
    --training.distributed.tp_size "$TP_SIZE"
    --training.distributed.hsdp_replicate_dim "$HSDP_REPLICATE_DIM"
    --training.distributed.hsdp_shard_dim "$HSDP_SHARD_DIM"
    --training.checkpoint.output_dir "$OUTPUT_DIR"
    --training.tracker.project_name "$PROJECT_NAME"
    --training.tracker.run_name "$RUN_NAME"
)

echo "DiffusionNFT Wan single-frame RL training config:"
echo "  config: $CONFIG"
echo "  data path: $DATA_PATH"
echo "  parquet files: $num_parquet"
echo "  output dir: $OUTPUT_DIR"
echo "  frames / latent T: $NUM_FRAMES / $NUM_LATENT_T"
echo "  rewards: pickscore + clipscore"
echo "  learning rate: 3e-5"
echo "  GPUs: $NUM_GPUS"
echo "  SP/TP: $SP_SIZE/$TP_SIZE"
echo "  HSDP replicate/shard: $HSDP_REPLICATE_DIM/$HSDP_SHARD_DIM"
echo "  W&B mode: $WANDB_MODE"
echo "  log file: $log_file"
echo "Command:"
printf '  %q' "${cmd[@]}" "$@"
echo

"${cmd[@]}" "$@" 2>&1 | tee "$log_file"
