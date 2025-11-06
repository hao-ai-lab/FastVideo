#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Prepare LongCat native weights directory with all components
#
# Usage:
#   ./prepare_longcat_native_weights.sh SOURCE_DIR OUTPUT_DIR
#
# Example:
#   ./prepare_longcat_native_weights.sh \
#     weights/longcat-for-fastvideo \
#     weights/longcat-native

set -e

SOURCE_DIR="$1"
OUTPUT_DIR="$2"

if [ -z "$SOURCE_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 SOURCE_DIR OUTPUT_DIR"
    echo ""
    echo "Example:"
    echo "  $0 weights/longcat-for-fastvideo weights/longcat-native"
    exit 1
fi

echo "==================================="
echo "Preparing LongCat Native Weights"
echo "==================================="
echo ""
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Convert transformer weights
echo "Step 1: Converting transformer weights..."
python scripts/checkpoint_conversion/longcat_native_weights_converter.py \
    --source "$SOURCE_DIR/transformer" \
    --output "$OUTPUT_DIR/transformer" \
    --validate

# Step 2: Copy other components
echo ""
echo "Step 2: Copying other components..."

# Copy tokenizer
if [ -d "$SOURCE_DIR/tokenizer" ]; then
    echo "  Copying tokenizer..."
    cp -r "$SOURCE_DIR/tokenizer" "$OUTPUT_DIR/"
fi

# Copy text_encoder
if [ -d "$SOURCE_DIR/text_encoder" ]; then
    echo "  Copying text_encoder..."
    cp -r "$SOURCE_DIR/text_encoder" "$OUTPUT_DIR/"
fi

# Copy VAE
if [ -d "$SOURCE_DIR/vae" ]; then
    echo "  Copying vae..."
    cp -r "$SOURCE_DIR/vae" "$OUTPUT_DIR/"
fi

# Copy scheduler
if [ -d "$SOURCE_DIR/scheduler" ]; then
    echo "  Copying scheduler..."
    cp -r "$SOURCE_DIR/scheduler" "$OUTPUT_DIR/"
fi

# Copy LoRA if exists
if [ -d "$SOURCE_DIR/lora" ]; then
    echo "  Copying lora (optional)..."
    cp -r "$SOURCE_DIR/lora" "$OUTPUT_DIR/"
fi

# Step 3: Create model_index.json for native model
echo ""
echo "Step 3: Creating model_index.json..."

cat > "$OUTPUT_DIR/model_index.json" << 'EOF'
{
  "_class_name": "LongCatPipeline",
  "_diffusers_version": "0.32.0",
  "workload_type": "video-generation",
  "tokenizer": [
    "transformers",
    "AutoTokenizer"
  ],
  "text_encoder": [
    "transformers",
    "UMT5EncoderModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKLWan"
  ],
  "scheduler": [
    "diffusers",
    "FlowMatchEulerDiscreteScheduler"
  ],
  "transformer": [
    "diffusers",
    "LongCatTransformer3DModel"
  ]
}
EOF

echo "  ✓ Created model_index.json (points to native LongCatTransformer3DModel)"

# Step 4: Summary
echo ""
echo "==================================="
echo "✓ Native weights prepared!"
echo "==================================="
echo ""
echo "Directory structure:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Test loading:"
echo "     python test_longcat_native.py"
echo ""
echo "  2. Run inference:"
echo "     from fastvideo import VideoGenerator"
echo "     generator = VideoGenerator.from_pretrained('$OUTPUT_DIR')"
echo "     video = generator.generate_video(prompt='A cat playing piano')"
echo ""

