#!/bin/bash
# Simple wrapper for convert_ltx2_weights.py
# Edit the values below directly

python /home/d1su/codes/FastVideo/scripts/checkpoint_conversion/convert_ltx2_weights.py \
    --source "<PATH_TO_LOCAL_REPO>/Lightricks/LTX-2/ltx-2-19b-dev.safetensors" \
    --output "converted_weights/ltx2-base" \
    --class-name "LTX2Transformer3DModel" \
    --pipeline-class-name "LTX2Pipeline" \
    --diffusers-version "0.33.0.dev0" \
    --gemma-path "<PATH_TO_LOCAL_REPO>/google/gemma-3-12b-it"
