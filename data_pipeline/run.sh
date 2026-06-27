#!/bin/bash
# Stage-0 data pipeline: generate videos (Wan2.2-14B T2V) -> extract tracks (CoTracker v3).
# Runs both steps on the shao_wm allocation via `srun --overlap`.
#
# You pick the GPUs (the node is shared); pass them explicitly:
#   CUDA_VISIBLE_DEVICES=2,3 bash data_pipeline/run.sh            # full run (defaults below)
#   CUDA_VISIBLE_DEVICES=2,3 bash data_pipeline/run.sh --smoke    # 2 videos
#   CUDA_VISIBLE_DEVICES=2,3 NUM_VIDEOS=50 OUTPUT_DIR=... bash data_pipeline/run.sh
#
# num_gpus is derived from CUDA_VISIBLE_DEVICES. Never run model code on the login node.
set -euo pipefail

REPO=/mnt/weka/home/hao.zhang/shao/FastVideo
PY="$REPO/.venv/bin/python"
PROMPTS="${PROMPTS:-$REPO/examples/dataset/vidprom/prompts/vidprom_filtered_extended.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_t2v_720p}"
JOB_NAME="${JOB_NAME:-shao_wm}"
NUM_VIDEOS="${NUM_VIDEOS:-50}"
GRID_SIZE="${GRID_SIZE:-50}"

[[ "${1:-}" == "--smoke" ]] && NUM_VIDEOS=2

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPUs to use, e.g. CUDA_VISIBLE_DEVICES=2,3}"
IFS=',' read -ra _gpus <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#_gpus[@]}}"

JOBID="${JOBID:-$(squeue -u "$USER" -n "$JOB_NAME" -h -o %i | head -1)}"
[[ -n "$JOBID" ]] || { echo "[run] ERROR: no running '$JOB_NAME' allocation" >&2; exit 1; }
echo "[run] jobid=$JOBID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  num_gpus=$NUM_GPUS  num_videos=$NUM_VIDEOS"

# /usr/bin/env (absolute: bare `env` may resolve to a broken ~/.local/bin/env) pins the
# device list inside the step. Not SLURM --export: its comma parsing splits "2,3".
SRUN=(srun --jobid="$JOBID" --overlap --ntasks=1 /usr/bin/env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")

echo "[run] === generate videos ==="
"${SRUN[@]}" "$PY" "$REPO/data_pipeline/generate_videos.py" \
  --prompts "$PROMPTS" --output-dir "$OUTPUT_DIR" --num-videos "$NUM_VIDEOS" --num-gpus "$NUM_GPUS"

echo "[run] === extract tracks ==="
"${SRUN[@]}" "$PY" "$REPO/data_pipeline/extract_tracks.py" \
  --data-dir "$OUTPUT_DIR" --grid-size "$GRID_SIZE"

echo "[run] done -> $OUTPUT_DIR"
