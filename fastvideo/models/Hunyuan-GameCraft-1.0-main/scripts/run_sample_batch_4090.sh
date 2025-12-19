#!/bin/bash
JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE="/path/to/models"
checkpoint_path="/path/to/ckpts"

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
modelname='Tencent_hunyuanGameCraft_720P'

# disable sp and enable cpu offload
export DISABLE_SP=1
export CPU_OFFLOAD=1
export NUM_GPU=1

# # enable both sp and cpu offload
# export DISABLE_SP=0
# export CPU_OFFLOAD=1
# export NUM_GPU=8

torchrun --nnodes=1 --nproc_per_node=${NUM_GPU} --master_port 29605 hymm_sp/sample_batch.py \
    --image-path "asset/village.png" \
    --prompt "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky." \
    --add-neg-prompt "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${checkpoint_path} \
    --video-size 704 1216 \
    --cfg-scale 2.0 \
    --image-start \
    --action-list w s d a \
    --action-speed-list 0.2 0.2 0.2 0.2 \
    --seed 250160 \
    --infer-steps 50 \
    --flow-shift-eval-video 5.0 \
    --cpu-offload \
    --use-fp8 \
    --save-path './results/'
