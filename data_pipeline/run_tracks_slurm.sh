#!/usr/bin/env bash
# Multi-node / multi-GPU data-parallel CoTracker on this Slinky GB200 cluster.
# Each Slurm task = one CoTracker worker (B=1); PROCS_PER_GPU tasks share each GPU.
# Runs inside the enroot `fvbuild` container (needed for the venv's CUDA runtime).
#
# Usage:
#   VIDEO_LIST=/mnt/lustre/vlm-s4duan/openvid/videos.txt \
#   OUT_DIR=/mnt/lustre/vlm-s4duan/openvid/tracks \
#   NODES=4 PROCS_PER_GPU=2 FPS=24 NUM_FRAMES=121 GRID=50 \
#   bash data_pipeline/run_tracks_slurm.sh
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
: "${VIDEO_LIST:?set VIDEO_LIST}"; : "${OUT_DIR:?set OUT_DIR}"
NODES=${NODES:-1}; GPUS=${GPUS:-4}; PROCS_PER_GPU=${PROCS_PER_GPU:-2}
FPS=${FPS:-24}; NUM_FRAMES=${NUM_FRAMES:-121}; GRID=${GRID:-50}
CLIPS_ARG=""; [ -n "${CLIPS_DIR:-}" ] && CLIPS_ARG="--clips-dir $CLIPS_DIR"
TASKS_PER_NODE=$(( GPUS * PROCS_PER_GPU ))
CPUS_PER_TASK=$(( 128 / TASKS_PER_NODE )); [ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1

mkdir -p "$OUT_DIR" "$WORK/logs"
echo "nodes=$NODES gpus/node=$GPUS procs/gpu=$PROCS_PER_GPU -> $((NODES*TASKS_PER_NODE)) workers"

# Bare-node: torch (self-contained cu128 wheels) + CoTracker need only the driver +
# the Lustre venv — NO container (avoids a per-node image pull).
sbatch -N "$NODES" --gres=gpu:$GPUS --ntasks-per-node=$TASKS_PER_NODE --exclusive \
  --cpus-per-task=$CPUS_PER_TASK --mem=0 -t 24:00:00 -J cotracker_dp \
  --chdir="$WORK/FastVideo" -o "$WORK/logs/cotracker_dp_%j.out" -e "$WORK/logs/cotracker_dp_%j.out" \
  --wrap "srun --chdir=$WORK/FastVideo \
    bash -lc 'source .venv/bin/activate && export TORCH_HOME=$WORK/.torch TOKENIZERS_PARALLELISM=false && \
      python data_pipeline/extract_tracks_mp.py \
        --video-list $VIDEO_LIST --out-dir $OUT_DIR $CLIPS_ARG \
        --gpus-per-node $GPUS --fps $FPS --num-frames $NUM_FRAMES --grid-size $GRID'"
