#!/bin/bash
# Run segment_tracks.py in parallel across 4 GPUs.
# Usage: bash data_pipeline/run_segment_tracks.sh [extra args]
# Example: bash data_pipeline/run_segment_tracks.sh --limit 20

DATA_DIR=/home/hal-kevin/data/motion-stream-test
WORLD_SIZE=4
LOG_FILE=data_pipeline/segment_tracks.log

> $LOG_FILE  # truncate on each run

echo "[seg] launching $WORLD_SIZE workers... logging to $LOG_FILE"

for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python -u data_pipeline/segment_tracks.py \
        --data-dir $DATA_DIR \
        --videos-subdir roundtrip_videos \
        --conf 0.75 --iou 0.9 --imgsz 1024 \
        --vis-override-every 3 --viz \
        --rank $RANK --world-size $WORLD_SIZE \
        "$@" \
        >> $LOG_FILE 2>&1 &
done

wait
echo "[seg] all done. log at $LOG_FILE"
