#!/bin/bash

# LongCat Image-to-Video (I2V) Inference Script
# 
# This script runs LongCat I2V inference using the fastvideo CLI.
# LongCat I2V takes an input image and generates a video from it.
#
# Usage:
#   bash scripts/inference/v1_inference_longcat_i2v.sh
#
# Prerequisites:
#   - Install fastvideo: pip install -e .
#   - The model weights will be auto-downloaded from HuggingFace
#   - Or use local weights if you have them

num_gpus=1

export FASTVIDEO_ATTENTION_BACKEND=

# Model path options:
# Option 1: HuggingFace model (auto-downloaded)
export MODEL_BASE=FastVideo/LongCat-Video-I2V-Diffusers

# Option 2: Local weights (uncomment if you have local weights)
# export MODEL_BASE=weights/longcat-for-i2v

# Input image path (must be square for LongCat I2V)
IMAGE_PATH="assets/girl.png"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image not found at $IMAGE_PATH"
    echo "Please provide a valid image path"
    exit 1
fi

fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload False \
    --vae-cpu-offload True \
    --text-encoder-cpu-offload True \
    --pin-cpu-memory False \
    --enable-bsa False \
    --image-path "$IMAGE_PATH" \
    --height 480 \
    --width 480 \
    --num-frames 93 \
    --num-inference-steps 50 \
    --fps 15 \
    --guidance-scale 4.0 \
    --prompt "A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --output-path outputs_video/longcat_i2v



