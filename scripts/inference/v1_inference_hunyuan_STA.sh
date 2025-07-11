#!/bin/bash

num_gpus=2
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_hunyuan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size ${num_gpus} \
    --tp-size 1 \
    --height 768 \
    --width 1280 \
    --num-frames 117 \
    --num-inference-steps 50 \
    --guidance-scale 1 \
    --embedded-cfg-scale 6 \
    --flow-shift 7 \
    --prompt "A beautiful woman in a red dress walking down a street" \
    --seed 1024 \
    --output-path outputs_video/
