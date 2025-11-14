#!/bin/bash

# LongCat Simple Refinement Script
# This script refines an existing 480p video to 720p
# Usage: bash scripts/inference/v1_inference_longcat_refine_simple.sh

# Number of GPUs
num_gpus=1

# Attention backend
export FASTVIDEO_ATTENTION_BACKEND=

# Model path
export MODEL_BASE=weights/longcat-native

# Input video (change this to your 480p video path)
INPUT_VIDEO="outputs_video/longcat_distill/source.mp4"

# Output directory
REFINE_OUTPUT="outputs_video/longcat_refine_720p"

# Prompt (should match the original prompt used for base generation)
PROMPT="In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."

NEGATIVE_PROMPT="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

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
    echo "Please set INPUT_VIDEO to your 480p video path"
    exit 1
fi

# ==============================================================================
# Refine 480p -> 720p with BSA and refinement LoRA
# ==============================================================================
echo "🔧 Configuring refinement (BSA enabled, refinement LoRA)..."
echo "✅ Input video: $INPUT_VIDEO"
echo "✅ BSA enabled with sparsity=0.9375"
echo "✅ Refinement LoRA loaded"
echo ""

fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload True \
    --vae-cpu-offload False \
    --text-encoder-cpu-offload True \
    --pin-cpu-memory False \
    --enable-bsa True \
    --bsa-sparsity 0.9375 \
    --bsa-chunk-q 4 4 8 \
    --bsa-chunk-k 4 4 8 \
    --lora-path "$MODEL_BASE/lora/refinement" \
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
    --negative-prompt "$NEGATIVE_PROMPT" \
    --seed 42 \
    --output-path "$REFINE_OUTPUT"

echo ""
echo "=========================================="
echo "✓ Refinement Complete!"
echo "=========================================="
echo ""
echo "Output directory: $REFINE_OUTPUT"
echo ""

