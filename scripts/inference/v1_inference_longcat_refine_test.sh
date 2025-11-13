#!/bin/bash

# LongCat 480p -> 720p Refinement Test Script
# This script demonstrates the full refinement pipeline:
# 1. Generate 480p base video (with distillation)
# 2. Refine to 720p with BSA and refinement LoRA

# Number of GPUs
num_gpus=1

# Attention backend
export FASTVIDEO_ATTENTION_BACKEND=

# Model path
export MODEL_BASE=weights/longcat-native

# Output directories
BASE_OUTPUT="outputs_video/longcat_base_480p"
REFINE_OUTPUT="outputs_video/longcat_refine_720p"

# Shared prompt
PROMPT="In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."

NEGATIVE_PROMPT="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

echo "=========================================="
echo "LongCat Refinement Pipeline"
echo "=========================================="
echo ""

# ==============================================================================
# Step 1: Generate 480p base video (16-step distillation)
# ==============================================================================
echo "Step 1: Generating 480p base video with distillation..."
echo ""

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
    --lora-path "$MODEL_BASE/lora/distilled" \
    --lora-nickname "distilled" \
    --height 480 \
    --width 832 \
    --num-frames 93 \
    --num-inference-steps 16 \
    --fps 15 \
    --guidance-scale 1.0 \
    --prompt "$PROMPT" \
    --negative-prompt "$NEGATIVE_PROMPT" \
    --seed 42 \
    --output-path "$BASE_OUTPUT"

# Find the generated video file
BASE_VIDEO=$(find "$BASE_OUTPUT" -name "*.mp4" -type f | head -n 1)

if [ -z "$BASE_VIDEO" ]; then
    echo "Error: Base video not found in $BASE_OUTPUT"
    exit 1
fi

echo ""
echo "✓ Base video generated: $BASE_VIDEO"
echo ""

# ==============================================================================
# Step 2: Refine 480p -> 720p with BSA and refinement LoRA
# ==============================================================================
echo "Step 2: Refining to 720p with BSA and refinement LoRA..."
echo ""

fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload False \
    --vae-cpu-offload True \
    --text-encoder-cpu-offload True \
    --pin-cpu-memory False \
    --enable-bsa True \
    --bsa-sparsity 0.9375 \
    --bsa-chunk-q 4 4 4 \
    --bsa-chunk-k 4 4 4 \
    --lora-path "$MODEL_BASE/lora/refinement" \
    --lora-nickname "refinement" \
    --refine-from "$BASE_VIDEO" \
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

REFINE_VIDEO=$(find "$REFINE_OUTPUT" -name "*.mp4" -type f | head -n 1)

echo ""
echo "=========================================="
echo "✓ Refinement Pipeline Complete!"
echo "=========================================="
echo ""
echo "Base video (480p):   $BASE_VIDEO"
echo "Refined video (720p): $REFINE_VIDEO"
echo ""

