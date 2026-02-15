#!/bin/bash
# Prepare video and prompts into video2caption.json for finetuning
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "$0")/../../../../.." && pwd)"
OVERFIT_DIR="$ROOT_DIR/examples/training/finetune/ltx2/overfit"
DATA_DIR="$OVERFIT_DIR/data"
VIDEOS_DIR="$DATA_DIR/generated_videos"
PROMPTS_JSONL="$ROOT_DIR/data/Davids048/ltx2-data/merged_prompts_semantic_unique_16k/video_prompts.gpt-5-mini-2025-08-07.0-1000.jsonl"
OUTPUT_JSON="$DATA_DIR/videos2caption.json"
WORKERS="${WORKERS:-1}"

if [ ! -d "$VIDEOS_DIR" ]; then
  echo "Missing videos directory: $VIDEOS_DIR" >&2
  exit 1
fi

if [ ! -f "$PROMPTS_JSONL" ]; then
  echo "Missing prompts jsonl: $PROMPTS_JSONL" >&2
  exit 1
fi

conda run --no-capture-output -n fastvideo python -u "$OVERFIT_DIR/build_videos2caption_from_jsonl.py" \
  --videos-dir "$VIDEOS_DIR" \
  --prompts-jsonl "$PROMPTS_JSONL" \
  --output-json "$OUTPUT_JSON" \
  --workers "$WORKERS"

# Preprocess merged dataset expects <dataset_root>/videos/ + videos2caption.json.
if [ ! -e "$DATA_DIR/videos" ]; then
  ln -s generated_videos "$DATA_DIR/videos"
  echo "Created symlink: $DATA_DIR/videos -> generated_videos"
fi

echo "Done: $OUTPUT_JSON"
