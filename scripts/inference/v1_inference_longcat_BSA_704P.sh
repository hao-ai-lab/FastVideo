#!/bin/bash

# LongCat inference script WITH BSA (Block Sparse Attention)
# Usage: bash scripts/inference/v1_inference_longcat_BSA.sh
#
# This script enables BSA for better performance on high-resolution generation.
# BSA reduces memory usage and increases speed with minimal quality loss.

# Number of GPUs
num_gpus=1

# Attention backend:
# - leave empty to auto-select best available
# - set to TORCH_SDPA for maximum compatibility
# - set to FLASH_ATTN if FlashAttention is installed and desired
export FASTVIDEO_ATTENTION_BACKEND=

# Model path:
# - Native (recommended after weight conversion)
export MODEL_BASE=weights/longcat-native
# - Or use the wrapper (Phase 1) for reference:

# ==============================================================================
# BSA Configuration
# ==============================================================================
# Automatically enable BSA before inference
echo "🔧 Configuring BSA (Block Sparse Attention)..."

# Check if config file exists
CONFIG_FILE="$MODEL_BASE/transformer/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    echo "   Please ensure your model directory is correct."
    exit 1
fi

# Enable BSA using original LongCat refinement parameters
# 704×1280×96, chunk [4,4,4], sparsity 0.9375
python scripts/checkpoint_conversion/manage_bsa.py "$CONFIG_FILE" \
    --enable \
    --sparsity 0.9375 \
    --chunk-q 4 4 4 \
    --chunk-k 4 4 4 \
    --no-backup

echo "✅ BSA enabled (original LongCat refinement config)"
echo "   - Resolution: 704×1280×96 frames"
echo "   - Latent shape after VAE+Patch: (96, 44, 80)"
echo "   - chunk_3d_shape: [4, 4, 4] (all divisible)"
echo "   - sparsity: 0.9375"
echo ""
# ==============================================================================

# You can either use --prompt or --prompt-txt, but not both.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload False \
    --vae-cpu-offload False \
    --text-encoder-cpu-offload True \
    --pin-cpu-memory False \
    --height 704 \
    --width 1280 \
    --num-frames 96 \
    --num-inference-steps 50 \
    --fps 15 \
    --guidance-scale 4.0 \
    --prompt "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --output-path outputs_video/longcat_bsa_704p

# Optional: provide an inline prompt instead of a file
# fastvideo generate \
#     --model-path $MODEL_BASE \
#     --sp-size $num_gpus \
#     --tp-size 1 \
#     --num-gpus $num_gpus \
#     --dit-cpu-offload False \
#     --vae-cpu-offload False \
#     --text-encoder-cpu-offload True \
#     --pin-cpu-memory False \
#     --height 480 \
#     --width 832 \
#     --num-frames 65 \
#     --num-inference-steps 50 \
#     --fps 16 \
#     --guidance-scale 4.0 \
#     --prompt "A cat playing piano, high quality, cinematic" \
#     --seed 1024 \
#     --output-path outputs_video/


