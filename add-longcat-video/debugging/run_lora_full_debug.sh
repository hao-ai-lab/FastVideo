#!/bin/bash
# Run complete LoRA debugging comparison

set -e  # Exit on error

echo "=========================================="
echo "Complete LoRA Debugging Comparison"
echo "=========================================="
echo ""

# Stage 1: Original model with longcat_shao env
echo "Stage 1: Running original LongCat (longcat_shao env)..."
echo "  - Original WITHOUT LoRA"
echo "  - Original WITH LoRA"
echo "------------------------------------------"
conda run -n longcat_shao python debug_lora_full_comparison.py --stage 1
echo ""
echo "✓ Stage 1 complete"
echo ""

# Stage 2: Native model with fastvideo_shao env  
echo "Stage 2: Running native FastVideo (fastvideo_shao env)..."
echo "  - Native WITHOUT LoRA"
echo "  - Native WITH LoRA (manually applied)"
echo "  - Analysis and comparison"
echo "------------------------------------------"
conda run -n fastvideo_shao python debug_lora_full_comparison.py --stage 2
echo ""
echo "✓ Stage 2 complete"
echo ""

echo "=========================================="
echo "✅ Complete comparison finished!"
echo "=========================================="
echo "Results saved in: outputs/debug_lora_comparison/"
echo "  - original_no_lora.pt"
echo "  - original_with_lora.pt"
echo "  - native_no_lora.pt"
echo "  - native_with_lora.pt"

