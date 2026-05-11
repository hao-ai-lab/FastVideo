#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${1:-train.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/train_text_only_dmd_preprocessed}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
GPU_NUM="${GPU_NUM:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-8}"
FLUSH_FREQUENCY="${FLUSH_FREQUENCY:-8}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-512}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fastvideo}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-22000}"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input text file not found: $INPUT_FILE" >&2
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

visible_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ "$visible_gpus" -lt "$GPU_NUM" ]]; then
    echo "Expected at least $GPU_NUM GPUs, found $visible_gpus" >&2
    exit 1
fi

for gpu_id in $(seq 0 $((GPU_NUM - 1))); do
    free_mb=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$free_mb" -lt "$MIN_FREE_GPU_MB" ]]; then
        echo "GPU $gpu_id has only ${free_mb} MiB free; text-only Wan preprocessing needs" \
            "about ${MIN_FREE_GPU_MB} MiB." >&2
        echo "Free the GPU or lower MIN_FREE_GPU_MB if you know this run will fit." >&2
        exit 1
    fi
done

MARKER="$OUTPUT_DIR/.fastvideo_text_only_dmd_output"
if [[ -d "$OUTPUT_DIR" && ! -f "$MARKER" ]]; then
    echo "Refusing to overwrite existing non-script output directory: $OUTPUT_DIR" >&2
    echo "Set OUTPUT_DIR to a new path or remove the directory manually." >&2
    exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
touch "$MARKER"

echo "Text-only DMD preprocessing config:"
echo "  input: $INPUT_FILE"
echo "  output: $OUTPUT_DIR"
echo "  model: $MODEL_PATH"
echo "  gpus: $GPU_NUM"
echo "  batch size per GPU: $BATCH_SIZE"
echo "  text max length: $TEXT_MAX_LENGTH"
echo "  samples per parquet file: $SAMPLES_PER_FILE"
echo "  flush frequency: $FLUSH_FREQUENCY"

SHARD_DIR="$OUTPUT_DIR/_text_shards"
mkdir -p "$SHARD_DIR"

python - "$INPUT_FILE" "$SHARD_DIR" "$GPU_NUM" <<'PY'
from pathlib import Path
import sys

input_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])

prompts = [line.rstrip("\n") for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not prompts:
    raise SystemExit(f"No non-empty prompts found in {input_path}")

for shard_idx in range(num_shards):
    shard_prompts = prompts[shard_idx::num_shards]
    shard_path = shard_dir / f"train_text_shard_{shard_idx}.txt"
    shard_path.write_text("\n".join(shard_prompts) + "\n", encoding="utf-8")
    print(f"Wrote {len(shard_prompts)} prompts to {shard_path}")
PY

run_preprocess_worker() {
    local gpu_id="$1"
    local shard_file="$2"
    local shard_output="$3"
    local log_file="$4"
    local master_port="$5"
    local -a cmd=(
        torchrun
        --nnodes=1
        --nproc_per_node=1
        --master_port "$master_port"
        fastvideo/pipelines/preprocess/v1_preprocess.py
        --model_path "$MODEL_PATH"
        --data_merge_path "$shard_file"
        --preprocess_video_batch_size "$BATCH_SIZE"
        --seed 42
        --max_height 448
        --max_width 832
        --num_frames 77
        --dataloader_num_workers 0
        --output_dir "$shard_output"
        --train_fps 16
        --samples_per_file "$SAMPLES_PER_FILE"
        --flush_frequency "$FLUSH_FREQUENCY"
        --text_max_length "$TEXT_MAX_LENGTH"
        --video_length_tolerance_range 5
        --preprocess_task text_only
    )

    {
        echo "[gpu${gpu_id}] log file: $log_file"
        echo "[gpu${gpu_id}] command: CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
    } | tee "$log_file"

    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 \
        | sed -u "s/^/[gpu${gpu_id}] /" \
        | tee -a "$log_file"
    local status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 ]]; then
        echo "[gpu${gpu_id}] preprocessing failed with exit code $status" | tee -a "$log_file"
    fi
    return "$status"
}

pids=()
for gpu_id in $(seq 0 $((GPU_NUM - 1))); do
    shard_file="$SHARD_DIR/train_text_shard_${gpu_id}.txt"
    shard_output="$OUTPUT_DIR/shard_${gpu_id}"
    mkdir -p "$shard_output"
    log_file="$OUTPUT_DIR/preprocess_gpu_${gpu_id}.log"

    echo "Launching text-only preprocessing on GPU ${gpu_id}: ${shard_file}"
    run_preprocess_worker "$gpu_id" "$shard_file" "$shard_output" "$log_file" "$((29610 + gpu_id))" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "One or more preprocessing workers failed. Check logs under $OUTPUT_DIR." >&2
    exit 1
fi

num_parquet=$(find "$OUTPUT_DIR" -name '*.parquet' | wc -l | tr -d ' ')
if [[ "$num_parquet" -eq 0 ]]; then
    echo "No parquet files were produced under $OUTPUT_DIR" >&2
    exit 1
fi

echo "Text-only preprocessing complete."
echo "Parquet files: $num_parquet"
echo "Use this training data_path:"
echo "$OUTPUT_DIR"
