#!/bin/bash
# Preprocess video+caption data for Cosmos 2.5 T2W training.
#
# This encodes:
#   - Videos → VAE latents via Cosmos25WanVAEWrapper (normalized internally)
#   - Captions → 100352-dim text embeddings via Reason1 (Qwen2.5-VL, full_concat)
#
# Output: parquet dataset ready for finetune_t2w.sh / finetune_t2w_lora.sh
#
# Input data format:
#   DATA_MERGE_PATH should point to a merge.txt with a single line:
#     "<video_dir>,<annotation.json>"
#   annotation.json format:
#     [{"path": "clip.mp4", "cap": ["caption"], "fps": 24.0, "duration": 3.2}]
#
#   See examples/training/finetune/wan_t2v_1.3B/crush_smol/ for the expected format.
#
# Hardware requirements:
#   - NVIDIA GPU with >= 24GB VRAM (VAE + Reason1 loaded simultaneously)
#   - Reason1 (Qwen2.5-VL-7B) requires ~15GB; VAE requires ~2GB
#
# Note: this script uses v1_preprocess.py. Port to v1_preprocessing_new is pending.
#
# Usage:
#   bash examples/training/finetune/cosmos2_5/preprocess_cosmos2_5_t2w.sh

GPU_NUM=1
MODEL_PATH="KyleShao/Cosmos-Predict2.5-2B-Diffusers"
DATA_MERGE_PATH="data/cosmos2_5_raw/merge.txt"
OUTPUT_DIR="data/cosmos2_5_processed_t2w/"

torchrun --nproc_per_node=$GPU_NUM \
    --master_port=29514 \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 1 \
    --seed 42 \
    --max_height 704 \
    --max_width 1280 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir $OUTPUT_DIR \
    --train_fps 24 \
    --samples_per_file 4 \
    --flush_frequency 4 \
    --video_length_tolerance_range 5 \
    --preprocess_task "t2v"