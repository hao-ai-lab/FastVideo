#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Preprocess HYWorld Training Data
# This script precomputes VAE latents and text embeddings for efficient training

set -e

# ============================================================================
# Configuration - Modify these paths for your setup
# ============================================================================

# Model paths
MODEL_PATH="/path/to/HunyuanVideo-1.5"
ACTION_CKPT="/path/to/HY-WorldPlay/ar_model"

# Input manifest (JSON with video paths and captions)
RAW_MANIFEST="/path/to/raw_dataset.json"

# Output directory for precomputed latents
OUTPUT_ROOT="./precomputed_latents"

# Processing options
MAX_SAMPLES=0  # 0 = process all, >0 = limit samples
DEVICE="cuda"
DTYPE="bf16"  # bf16 or fp16

# ============================================================================
# Run Preprocessing
# ============================================================================

echo "Starting HYWorld data preprocessing..."
echo "  Model: $MODEL_PATH"
echo "  Input: $RAW_MANIFEST"
echo "  Output: $OUTPUT_ROOT"

python -m fastvideo.dataset.preprocess_hyworld \
    --raw-manifest "$RAW_MANIFEST" \
    --model-path "$MODEL_PATH" \
    --action-ckpt "$ACTION_CKPT" \
    --out-root "$OUTPUT_ROOT" \
    --max-samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --dtype "$DTYPE"

echo "Preprocessing complete!"
echo ""
echo "Next steps:"
echo "1. Create a training manifest JSON pointing to the precomputed latents"
echo "2. Each entry should have:"
echo "   - latent_path: path to the .pt file with precomputed latents"
echo "   - pose_path: path to the camera pose JSON"
echo "   - (optional) action_path: path to action labels JSON"
echo ""
echo "Example manifest entry:"
echo '{'
echo '  "latent_path": "./precomputed_latents/sample_00000/latent.pt",'
echo '  "pose_path": "/path/to/video_poses.json"'
echo '}'
