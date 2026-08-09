#!/bin/bash
# Single-sample VSA-H3 DMD2 overfit for MiniMax H3 (sparsity 0.0, parity vs
# dense). Two steps:
#   (a) preprocess ONE mp4 (with audio) + prompt into the t2va parquet row
#       the H3 dataloader consumes (768x1344, 124 frames @ 24 fps, 32 kHz);
#   (b) train on 4 GPUs: student on VIDEO_SPARSE_ATTN_H3, teacher/critic on
#       FLASH_ATTN with the FA4 path (FASTVIDEO_FA4=1).
#
# Usage:
#   VIDEO=/path/clip.mp4 PROMPT="a caption" bash examples/distill/MiniMax-H3/overfit_vsa0.sh
#   SKIP_PREPROCESS=1 bash examples/distill/MiniMax-H3/overfit_vsa0.sh   # data already written

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=${MASTER_PORT:-29513}

NUM_GPUS=${NUM_GPUS:-4}
CONFIG=${CONFIG:-examples/train/configs/distribution_matching/minimax_h3/dmd2_vsa0_overfit.yaml}
DATA_DIR=${DATA_DIR:-/mnt/h3-dmd2-overfit/data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/minimax_h3_dmd2_vsa0_overfit}
MODEL_PATH=${MODEL_PATH:-data/models/MiniMax-H3}

# ---------------------------------------------------------------- preprocess
if [ -z "${SKIP_PREPROCESS:-}" ]; then
  VIDEO=${VIDEO:?set VIDEO=/path/to/clip.mp4 (with soundtrack) or SKIP_PREPROCESS=1}
  PROMPT=${PROMPT:?set PROMPT="text prompt for the clip"}
  python -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit \
    --video "$VIDEO" \
    --prompt "$PROMPT" \
    --output-dir "$DATA_DIR" \
    --model-path "$MODEL_PATH"
fi

# --------------------------------------------------------------------- train
# FA4 flash-attention path for the dense teacher/critic FLASH_ATTN roles.
export FASTVIDEO_FA4=1

torchrun \
  --nnodes 1 \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NUM_GPUS" \
  -m fastvideo.train.entrypoint.train \
  --config "$CONFIG" \
  --training.distributed.num_gpus "$NUM_GPUS" \
  --training.distributed.sp_size "$NUM_GPUS" \
  --training.distributed.hsdp_shard_dim "$NUM_GPUS" \
  --training.data.data_path "$DATA_DIR" \
  --training.checkpoint.output_dir "$OUTPUT_DIR"
