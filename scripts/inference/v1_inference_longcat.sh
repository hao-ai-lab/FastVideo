#!/bin/bash

# LongCat Text-to-Video (T2V) Inference Script
# 
# This script runs LongCat T2V inference using the fastvideo CLI.
#
# Usage:
#   bash scripts/inference/v1_inference_longcat.sh
#
# Prerequisites:
#   - Install fastvideo: pip install -e .
#   - The model weights will be auto-downloaded from HuggingFace

num_gpus=1

export FASTVIDEO_ATTENTION_BACKEND=

# Model path options:
# Option 1: HuggingFace model (auto-downloaded)
export MODEL_BASE=FastVideo/LongCat-Video-T2V-Diffusers

# Option 2: Local weights (uncomment if you have local weights)
# For local weights, convert the official weights to FastVideo native format
# conversion method: python scripts/checkpoint_conversion/longcat_to_fastvideo.py
# --source /path/to/LongCat-Video/weights/LongCat-Video
# --output weights/longcat-native
# export MODEL_BASE=weights/longcat-native

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
    --prompt "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --output-path outputs_video/longcat_t2v
