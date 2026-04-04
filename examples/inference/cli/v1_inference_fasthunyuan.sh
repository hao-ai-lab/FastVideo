#!/bin/bash

num_gpus=4
export MODEL_BASE=FastVideo/FastHunyuan-Diffusers
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --height 720 \
    --width 1280 \
    --num-frames 125 \
    --num-inference-steps 6 \
    --guidance-scale 1 \
    --embedded-cfg-scale 6 \
    --flow-shift 17 \
    --prompt "A beautiful woman in a red dress walking down a street" \
    --seed 1024 \
    --output-path outputs_video/