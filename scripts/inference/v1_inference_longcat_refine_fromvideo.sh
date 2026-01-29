#!/bin/bash

# LongCat T2V Refinement Script (480p -> 720p)
# 
# This script refines a 480p distilled video to 720p using the refinement LoRA.
# Run v1_inference_longcat_distill.sh first to generate the 480p video.
#
# Usage:
#   bash scripts/inference/v1_inference_longcat_refine_fromvideo.sh
#
# Prerequisites:
#   - Install fastvideo: pip install -e .
#   - The model weights will be auto-downloaded from HuggingFace
#   - Run v1_inference_longcat_distill.sh first to generate input video

num_gpus=1

export FASTVIDEO_ATTENTION_BACKEND=

# Model path - HuggingFace model (auto-downloaded)
export MODEL_BASE=FastVideo/LongCat-Video-T2V-Diffusers

INPUT_VIDEO="outputs_video/longcat_distill/In a realistic photography style, a white boy around seven or eight years old sits on a park bench,.mp4"
REFINE_OUTPUT="outputs_video/longcat_refine_720p"

# Prompt used for base generation (must match distill script)
PROMPT="In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."

echo "=========================================="
echo "LongCat 480p -> 720p Refinement"
echo "=========================================="
echo ""
echo "Input:  $INPUT_VIDEO"
echo "Output: $REFINE_OUTPUT"
echo ""

# Check if input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video not found: $INPUT_VIDEO"
    echo "Please run v1_inference_longcat_distill.sh first to generate the 480p video"
    exit 1
fi

echo "Configuring refinement (BSA enabled, refinement LoRA)..."
echo "Input video: $INPUT_VIDEO"
echo "BSA enabled with sparsity=0.875"
echo "Refinement LoRA loaded"
echo ""

fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload True \
    --vae-cpu-offload True \
    --text-encoder-cpu-offload True \
    --pin-cpu-memory False \
    --enable-bsa True \
    --bsa-sparsity 0.875 \
    --bsa-chunk-q 4 4 8 \
    --bsa-chunk-k 4 4 8 \
    --lora-path "FastVideo/LongCat-Video-T2V-Refinement-LoRA" \
    --lora-nickname "refinement" \
    --refine-from "$INPUT_VIDEO" \
    --t-thresh 0.5 \
    --spatial-refine-only False \
    --num-cond-frames 0 \
    --height 720 \
    --width 1280 \
    --num-inference-steps 50 \
    --fps 30 \
    --guidance-scale 1.0 \
    --prompt "$PROMPT" \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --output-path "$REFINE_OUTPUT"

echo ""
echo "=========================================="
echo "Refinement Complete!"
echo "=========================================="
echo ""
echo "Output directory: $REFINE_OUTPUT"
echo ""

