#!/bin/bash

# LongCat Video Continuation (VC) Inference Script
# 
# This script runs LongCat VC inference using the fastvideo CLI.
# LongCat VC takes an input video and generates a continuation of it.
#
# Usage:
#   bash scripts/inference/v1_inference_longcat_vc.sh
#
# Prerequisites:
#   - Install fastvideo: pip install -e .
#   - The model weights will be auto-downloaded from HuggingFace
#   - Or use local weights if you have them

num_gpus=1

export FASTVIDEO_ATTENTION_BACKEND=

# Model path options:
# Option 1: HuggingFace model (auto-downloaded)
export MODEL_BASE=FastVideo/LongCat-Video-VC-Diffusers

# Option 2: Local weights (uncomment if you have local weights)
# export MODEL_BASE=weights/longcat-vc-upload

# Input video path
VIDEO_PATH="assets/motorcycle.mp4"

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video not found at $VIDEO_PATH"
    echo "Please provide a valid video path"
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
    --video-path "$VIDEO_PATH" \
    --num-cond-frames 13 \
    --height 480 \
    --width 832 \
    --num-frames 93 \
    --num-inference-steps 50 \
    --fps 15 \
    --guidance-scale 4.0 \
    --prompt "A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides. The video captures the journey from the rider's perspective, emphasizing the sense of motion and adventure." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --output-path outputs_video/longcat_vc



