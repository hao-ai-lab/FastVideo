#!/bin/bash

# LongCat T2V Distilled Inference Script
# 
# This script runs LongCat T2V with distillation LoRA (16 steps instead of 50).
# Uses the distilled LoRA for faster generation.
#
# Usage:
#   bash scripts/inference/v1_inference_longcat_distill.sh
#
# Prerequisites:
#   - Install fastvideo: pip install -e .
#   - The model weights will be auto-downloaded from HuggingFace

num_gpus=1

export FASTVIDEO_ATTENTION_BACKEND=

# Model path - HuggingFace model (auto-downloaded)
export MODEL_BASE=FastVideo/LongCat-Video-T2V-Diffusers

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
    --lora-path "FastVideo/LongCat-Video-T2V-Distilled-LoRA" \
    --lora-nickname "distilled" \
    --height 480 \
    --width 832 \
    --num-frames 93 \
    --num-inference-steps 16 \
    --fps 15 \
    --guidance-scale 1.0 \
    --prompt "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --output-path outputs_video/longcat_distill
