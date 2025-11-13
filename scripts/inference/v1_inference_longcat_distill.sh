#!/bin/bash

# LongCat distilled inference (480p, 16 steps, LoRA)
# Usage: bash scripts/inference/v1_inference_longcat_distill.sh
# - Disables BSA for 480p
# - Applies distilled LoRA adapter
# - Sets 16 steps, guidance=1.0, no negative prompt

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
# BSA Configuration (Runtime via CLI)
# ==============================================================================
# Disable BSA via CLI for distilled (480p)
echo "🔧 Configuring distilled inference (BSA disabled via CLI for 480p)..."
echo "✅ BSA disabled - using standard attention for 480p distilled mode"
echo "💡 Tip: For refinement (720p+), use v1_inference_longcat_refine.sh"
# ==============================================================================

# You can either use --prompt or --prompt-txt, but not both.
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
    --prompt "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene." \
    --seed 42 \
    --output-path outputs_video/longcat_distill
