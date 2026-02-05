#!/bin/bash

GPU_NUM=1
MODEL_PATH="/home/d1su/codes/FastVideo/converted_weights/ltx2-base"
# DATASET_PATH="data/overfit"
DATASET_PATH="data/crush-smol"
OUTPUT_DIR="$DATASET_PATH"
WITH_AUDIO=true

# Convert one-file overfit metadata into merged format if needed.
if [ ! -f "$DATASET_PATH/videos2caption.json" ] && [ -f "$DATASET_PATH/overfit.json" ]; then
  python scripts/dataset_preparation/convert_to_merged_dataset.py \
    --items-json "$DATASET_PATH/overfit.json" \
    --output-dir "$DATASET_PATH"
fi


torchrun --nproc_per_node=$GPU_NUM \
    --master_port=29513 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.with_audio $WITH_AUDIO \
    --preprocess.preprocess_video_batch_size 1 \
    --preprocess.dataloader_num_workers 0 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 73 \
    --preprocess.train_fps 16 \
    --preprocess.video_length_tolerance_range 5
