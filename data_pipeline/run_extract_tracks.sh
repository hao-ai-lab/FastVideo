#!/bin/bash
# Run extract_tracks.py in parallel across 4 GPUs.
# Usage: bash data_pipeline/run_extract_tracks.sh [extra args]

DATA_DIR=/home/hal-kevin/data/motion-stream-test
WORLD_SIZE=4
LOG_FILE=data_pipeline/extract_tracks.log

> $LOG_FILE  # truncate on each run

echo "[track] launching $WORLD_SIZE workers... logging to $LOG_FILE"

for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python -u data_pipeline/extract_tracks.py \
        --data-dir $DATA_DIR \
        --grid-size 50 \
        --device cuda \
        --detect-entries \
        --sam-conf 0.75 \
        --sam-iou 0.9 \
        --sam-imgsz 1024 \
        --entry-sample-every 2 \
        --entry-min-area 0.001 \
        --entry-new-area 0.5 \
        --rank $RANK --world-size $WORLD_SIZE \
        "$@" \
        >> $LOG_FILE 2>&1 &
done

wait
echo "[track] all done. log at $LOG_FILE"
