#!/bin/bash
# Run all three steps of the LoRA debugging comparison

set -e  # Exit on error

echo "=========================================="
echo "LoRA Debugging Comparison"
echo "=========================================="
echo ""

# Step 1: Run original LongCat with LoRA (longcat_shao env)
echo "Step 1: Running original LongCat with LoRA (longcat_shao env)..."
echo "------------------------------------------"
conda run -n longcat_shao python debug_lora_step1_original.py
echo ""
echo "✓ Step 1 complete"
echo ""

# Step 2: Run native FastVideo with LoRA (fastvideo_shao env)
echo "Step 2: Running native FastVideo with LoRA (fastvideo_shao env)..."
echo "------------------------------------------"
conda run -n fastvideo_shao python debug_lora_step2_native.py
echo ""
echo "✓ Step 2 complete"
echo ""

# Step 3: Analyze results (can use either env, using fastvideo_shao)
echo "Step 3: Analyzing results..."
echo "------------------------------------------"
conda run -n fastvideo_shao python debug_lora_step3_analyze.py
echo ""
echo "✓ Step 3 complete"
echo ""

echo "=========================================="
echo "✅ All steps complete!"
echo "=========================================="
echo "Results saved in: outputs/debug_lora_comparison/"
echo "  - test_inputs.pt"
echo "  - original_activations.pt"
echo "  - native_activations.pt"
echo "  - lora_comparison_report.txt"

