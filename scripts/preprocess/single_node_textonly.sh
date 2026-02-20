#!/bin/bash

set -euo pipefail

# Single-node text-only preprocessing without Slurm or torchrun.
# Phase 1 merges shard files for all GPUs.
# Phase 2 launches exactly one preprocess job per GPU only if all merges succeed.


base_port=29603

export MODEL_BASE="${MODEL_BASE:-Davids048/LTX2-Base-Diffusers}"
PROMPTS_DIR="${PROMPTS_DIR:-data/ltx2-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/test-text-preprocessing/single_node}"

# File index range, inclusive.
START_FILE="${START_FILE:-0}"
END_FILE="${END_FILE:-999}"

# Comma-separated GPU list, e.g. "0,1,2,3,4,5,6,7" or "0,1"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_LIST is empty."
    exit 1
fi

if [ "$END_FILE" -lt "$START_FILE" ]; then
    echo "END_FILE ($END_FILE) must be >= START_FILE ($START_FILE)."
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"


echo "Running text-only preprocessing on a single node"
echo "MODEL_BASE: $MODEL_BASE"
echo "PROMPTS_DIR: $PROMPTS_DIR"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "START_FILE: $START_FILE"
echo "END_FILE: $END_FILE"
echo "GPU_LIST: $GPU_LIST"

pids=()
num_gpus="${#GPU_IDS[@]}"
launched=0

MERGED_DIR="${OUTPUT_ROOT}/merged_prompts"
mkdir -p "$MERGED_DIR"

# If enabled and all merged shard files already exist and are non-empty,
# skip merge and directly launch preprocess jobs.
ALLOW_SKIP_MERGE="${ALLOW_SKIP_MERGE:-1}"
merge_paths=()
output_dirs=()
all_merged_ready=1

for idx in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$idx]}"
    merged_path="${MERGED_DIR}/gpu_${gpu}_merged.txt"
    output_dir="${OUTPUT_ROOT}/GPU_${gpu}"
    merge_paths+=("$merged_path")
    output_dirs+=("$output_dir")
    if [ ! -s "$merged_path" ]; then
        all_merged_ready=0
    fi
done

if [ "$ALLOW_SKIP_MERGE" = "1" ] && [ "$all_merged_ready" -eq 1 ]; then
    echo "All merged shard txt files already exist. Skipping merge phase."
else
    # Phase 1: merge all shards first. If any shard merge fails, abort before launching.
    merge_failed=0
    merge_paths=()
    output_dirs=()

    for idx in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$idx]}"
        merged_path="${MERGED_DIR}/gpu_${gpu}_merged.txt"
        output_dir="${OUTPUT_ROOT}/GPU_${gpu}"
        : > "$merged_path"
        shard_file_count=0

        for file_num in $(seq "$((START_FILE + idx))" "$num_gpus" "$END_FILE"); do
            src="${PROMPTS_DIR}/video_prompt_row_${file_num}.json"
            if [ ! -f "$src" ]; then
                echo "Merge failed: missing input $src"
                merge_failed=1
                break
            fi
            cat "$src" | python -c "import json, sys; d=json.load(sys.stdin); print(d['video_prompt'])" >> "$merged_path"
            shard_file_count=$((shard_file_count + 1))
        done

        if [ "$merge_failed" -ne 0 ]; then
            break
        fi

        if [ "$shard_file_count" -eq 0 ]; then
            echo "Merge failed: GPU ${gpu} shard has no files."
            merge_failed=1
            break
        fi

        if [ ! -s "$merged_path" ]; then
            echo "Merge failed: merged file is empty for GPU ${gpu}: ${merged_path}"
            merge_failed=1
            break
        fi

        echo "GPU ${gpu}: merged ${shard_file_count} files -> ${merged_path}"
        merge_paths+=("$merged_path")
        output_dirs+=("$output_dir")
    done

    if [ "$merge_failed" -ne 0 ]; then
        echo "Aborting: not all shard merges succeeded. No GPU jobs were launched."
        exit 1
    fi

    echo "All shard merges succeeded. Launching GPU jobs."
fi

# Phase 2: launch exactly one preprocess job per GPU.
for idx in "${!GPU_IDS[@]}"; do
    port=$((base_port + idx))
    gpu="${GPU_IDS[$idx]}"
    merged_path="${merge_paths[$idx]}"
    output_dir="${output_dirs[$idx]}"

    if [ -z "${merged_path:-}" ] || [ -z "${output_dir:-}" ]; then
        echo "Aborting: missing launch metadata for GPU ${gpu}."
        exit 1
    fi

    mkdir -p "$output_dir"
    echo "GPU ${gpu}: launch one preprocess run -> ${output_dir}"
    echo "Command: CUDA_VISIBLE_DEVICES=${gpu} torchrun --nnodes=1 --nproc_per_node=1 --master_port ${port} fastvideo/pipelines/preprocess/v1_preprocess.py --model_path \"$MODEL_BASE\" --data_merge_path \"$merged_path\" --preprocess_video_batch_size 2 --seed 42 --max_height 1088 --max_width 1920 --num_frames 121 --dataloader_num_workers 0 --output_dir \"$output_dir\" --train_fps 24 --samples_per_file 125 --flush_frequency 25 --video_length_tolerance_range 5 --preprocess_task \"text_only\""
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nnodes=1 --nproc_per_node=1 --master_port $port \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
            --model_path "$MODEL_BASE" \
            --data_merge_path "$merged_path" \
            --preprocess_video_batch_size 1 \
            --seed 42 \
            --max_height 1088 \
            --max_width 1920 \
            --num_frames 121 \
            --dataloader_num_workers 0 \
            --output_dir "$output_dir" \
            --train_fps 24 \
            --samples_per_file 125 \
            --flush_frequency 25 \
            --video_length_tolerance_range 5 \
            --preprocess_task "text_only" &

    pids+=("$!")
    launched=$((launched + 1))
done

if [ "$launched" -eq 0 ]; then
    echo "No jobs launched."
    exit 1
fi

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "Some preprocessing jobs failed."
    exit 1
fi

echo "All text-only preprocessing jobs completed."