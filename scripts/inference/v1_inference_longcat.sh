#!/bin/bash

num_gpus=2

export FASTVIDEO_ATTENTION_BACKEND=

# For longcat, we must first convert the official weights to FastVideo native format
# conversion method: python scripts/checkpoint_conversion/longcat_to_fastvideo.py
# --source /path/to/LongCat-Video/weights/LongCat-Video
# --output weights/longcat-native
export MODEL_BASE=weights/longcat-native

fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload False \
    --vae-cpu-offload False \
    --text-encoder-cpu-offload False \
    --pin-cpu-memory False \
    --enable-bsa False \
    --height 480 \
    --width 832 \
    --num-frames 93 \
    --num-inference-steps 50 \
    --fps 15 \
    --guidance-scale 4.0 \
    --prompt-txt assets/prompt.txt \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --output-path outputs_video/longcat_480p
