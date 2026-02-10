#!/usr/bin/env bash
set -euo pipefail

# Edit these values and commit changes as needed.
PROMPT_DIR="/home/hal-jundas/codes/FastVideo-demo/data/Davids048/ltx2-data/video_prompts.gpt-5-mini-2025-08-07.0-1000"
OUTPUT_DIR="/home/hal-jundas/codes/FastVideo-demo/data/ltx2-base-videos/"
START_IDX=${1:-0}
END_IDX=${2:-4}
NUM_GPUS=4

python3 /home/hal-jundas/codes/FastVideo-demo/data_preperation/generate_videos_from_json_prompts.py \
  --prompt-dir "$PROMPT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --start-idx "$START_IDX" \
  --end-idx "$END_IDX" \
  --num-gpus "$NUM_GPUS" \
  --no-skip-existing
