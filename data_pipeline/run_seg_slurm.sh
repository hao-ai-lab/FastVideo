#!/usr/bin/env bash
# Multi-node data-parallel frame-0 segmentation (SAM2.1-b+) over OpenVid-1M.
# Bare node (torch-only ultralytics needs just the driver + Lustre venv, no container).
# Each Slurm task = one SAM worker; PROCS_PER_GPU tasks share each GPU.
#
# Usage:
#   DATA_DIR=/mnt/lustre/vlm-s4duan/openvid_1m NODES=6 PROCS_PER_GPU=2 \
#   bash data_pipeline/run_seg_slurm.sh
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
DATA_DIR=${DATA_DIR:-$WORK/openvid_1m}
NODES=${NODES:-1}; GPUS=${GPUS:-4}; PROCS_PER_GPU=${PROCS_PER_GPU:-2}
MODEL=${MODEL:-sam2.1_b.pt}; VIDEOS_SUBDIR=${VIDEOS_SUBDIR:-clips}
TASKS_PER_NODE=$(( GPUS * PROCS_PER_GPU ))
CPUS_PER_TASK=$(( 128 / TASKS_PER_NODE )); [ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1
TIME=${TIME:-12:00:00}
mkdir -p "$WORK/logs"
echo "nodes=$NODES gpus/node=$GPUS procs/gpu=$PROCS_PER_GPU -> $((NODES*TASKS_PER_NODE)) workers, model=$MODEL"

sbatch -N "$NODES" --gres=gpu:$GPUS --ntasks-per-node=$TASKS_PER_NODE --exclusive \
  --cpus-per-task=$CPUS_PER_TASK --mem=0 -t "$TIME" -J seg_sam_dp \
  --chdir="$WORK/FastVideo" -o "$WORK/logs/seg_sam_dp_%j.out" -e "$WORK/logs/seg_sam_dp_%j.out" \
  --wrap "srun --chdir=$WORK/FastVideo \
    bash -lc 'source .venv/bin/activate && \
      export TORCH_HOME=$WORK/.torch HF_HOME=$WORK/.hf MPLCONFIGDIR=$WORK/.mpl \
        YOLO_CONFIG_DIR=$WORK/.ultralytics TOKENIZERS_PARALLELISM=false \
        PYTHONPATH=$WORK/FastVideo:$WORK/FastVideo/data_pipeline && \
      python data_pipeline/segment_tracks_mp.py \
        --data-dir $DATA_DIR --videos-subdir $VIDEOS_SUBDIR --model $MODEL --gpus-per-node $GPUS'"
