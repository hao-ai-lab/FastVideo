#!/usr/bin/env bash
# CoTracker extraction for our synth mp4s. Same pattern as run_tracks_slurm.sh but
# with the aspect/lowres filters disabled (synth videos are exactly 720x1280) and
# H/W set to our training resolution 480x832.
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
: "${VIDEO_LIST:?}"; : "${OUT_DIR:?}"
NODES=${NODES:-4}; GPUS=${GPUS:-4}; PROCS_PER_GPU=${PROCS_PER_GPU:-2}
FPS=${FPS:-24}; NUM_FRAMES=${NUM_FRAMES:-121}; GRID=${GRID:-50}
HEIGHT=${HEIGHT:-480}; WIDTH=${WIDTH:-832}
TASKS_PER_NODE=$(( GPUS * PROCS_PER_GPU ))
CPUS_PER_TASK=$(( 128 / TASKS_PER_NODE )); [ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1

mkdir -p "$OUT_DIR" "$WORK/logs"
echo "nodes=$NODES gpus/node=$GPUS procs/gpu=$PROCS_PER_GPU -> $((NODES*TASKS_PER_NODE)) workers at ${HEIGHT}x${WIDTH}"

sbatch -N "$NODES" --gres=gpu:$GPUS --ntasks-per-node=$TASKS_PER_NODE --exclusive \
  --cpus-per-task=$CPUS_PER_TASK --mem=0 -t 6:00:00 -J cotracker_synth \
  --chdir="$WORK/FastVideo" -o "$WORK/logs/cotracker_synth_%j.out" -e "$WORK/logs/cotracker_synth_%j.out" \
  --wrap "srun --chdir=$WORK/FastVideo \
    bash -lc 'source .venv/bin/activate && export TORCH_HOME=$WORK/.torch TOKENIZERS_PARALLELISM=false && \
      python data_pipeline/extract_tracks_mp.py \
        --video-list $VIDEO_LIST --out-dir $OUT_DIR \
        --gpus-per-node $GPUS --fps $FPS --num-frames $NUM_FRAMES --grid-size $GRID \
        --height $HEIGHT --width $WIDTH --min-height 0 --aspect-tol 10.0'"
