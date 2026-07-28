#!/usr/bin/env bash
# SAM 2.1-b+ segmentation of frame-0 for our synth mp4s. Adds object_ids +
# track_weights to each tracks .npz. Idempotent (skips npz already labeled).
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
: "${DATA_DIR:?}"
NODES=${NODES:-4}; GPUS=${GPUS:-4}; PROCS_PER_GPU=${PROCS_PER_GPU:-2}
TASKS_PER_NODE=$(( GPUS * PROCS_PER_GPU ))
CPUS_PER_TASK=$(( 128 / TASKS_PER_NODE )); [ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1
NUM_SHARDS=$(( NODES * TASKS_PER_NODE ))

mkdir -p "$WORK/logs"
echo "nodes=$NODES gpus/node=$GPUS procs/gpu=$PROCS_PER_GPU -> $NUM_SHARDS workers"

sbatch -N "$NODES" --gres=gpu:$GPUS --ntasks-per-node=$TASKS_PER_NODE --exclusive \
  --cpus-per-task=$CPUS_PER_TASK --mem=0 -t 4:00:00 -J segment_synth \
  --chdir="$WORK/FastVideo" -o "$WORK/logs/segment_synth_%j.out" -e "$WORK/logs/segment_synth_%j.out" \
  --wrap "srun --chdir=$WORK/FastVideo \
    bash -lc 'source .venv/bin/activate && \
      export TORCH_HOME=$WORK/.torch HF_HOME=$WORK/.hf HOME=$WORK MPLCONFIGDIR=$WORK/.mpl \
        YOLO_CONFIG_DIR=$WORK/.ultralytics TOKENIZERS_PARALLELISM=false && \
      export CUDA_VISIBLE_DEVICES=\$(( SLURM_LOCALID % $GPUS )) && \
      python data_pipeline/segment_tracks.py \
        --data-dir $DATA_DIR \
        --num-shards $NUM_SHARDS --shard \$SLURM_PROCID'"
