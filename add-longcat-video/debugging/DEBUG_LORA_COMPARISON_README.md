# LoRA Debugging Comparison

This document explains the LoRA debugging comparison script and how to use it to investigate discrepancies between the original LongCat and native FastVideo implementations with LoRA.

## Problem Statement

The distilled LoRA generation (`output_t2v_distill.mp4`) appears worse than the 16-step normal generation (`output_t2v_16steps.mp4`), suggesting there may be an issue with how LoRA is being applied in the native FastVideo implementation.

## Debugging Approach

We run **4 comparisons** to isolate the issue:

1. **Original WITHOUT LoRA** vs **Native WITHOUT LoRA**
   - Verifies the base model weights were correctly converted
   - Should have very low MSE (< 1e-4)

2. **Original WITH LoRA** vs **Native WITH LoRA**
   - Tests if LoRA is applied identically in both implementations
   - This is where we expect to find the divergence

3. **Effect of LoRA on Original Model**
   - Shows how much LoRA changes the original model's output
   - Helps quantify expected LoRA impact

4. **Effect of LoRA on Native Model**
   - Shows how much LoRA changes the native model's output
   - Should be similar to the original's LoRA effect

## Files

### Main Script
- `debug_lora_full_comparison.py` - Complete comparison script

### Helper Scripts
- `run_lora_full_debug.sh` - Bash script to run both stages automatically

### Output Files (in `outputs/debug_lora_comparison/`)
- `test_inputs.pt` - Shared test inputs for both models
- `original_no_lora.pt` - Original model activations without LoRA
- `original_with_lora.pt` - Original model activations with LoRA
- `native_no_lora.pt` - Native model activations without LoRA
- `native_with_lora.pt` - Native model activations with LoRA

## Usage

### Option 1: Run Both Stages Automatically

```bash
./run_lora_full_debug.sh
```

This will:
1. Run stage 1 with `longcat_shao` environment
2. Run stage 2 with `fastvideo_shao` environment
3. Show analysis results

### Option 2: Run Stages Manually

**Stage 1 (longcat_shao environment):**
```bash
conda activate longcat_shao
python debug_lora_full_comparison.py --stage 1
```

**Stage 2 (fastvideo_shao environment):**
```bash
conda activate fastvideo_shao
python debug_lora_full_comparison.py --stage 2
```

## What the Script Does

### Stage 1: Original Model (longcat_shao env)

1. Loads the original LongCat model from `/mnt/fast-disks/hao_lab/shao/LongCat-Video/`
2. Creates identical test inputs (small latents for fast testing)
3. Runs two forward passes:
   - WITHOUT LoRA
   - WITH LoRA (`cfg_step_lora.safetensors`)
4. Captures intermediate activations at various layers
5. Saves all results to disk

### Stage 2: Native Model (fastvideo_shao env)

1. Loads the saved test inputs from stage 1
2. Uses `VideoGenerator.from_pretrained()` (same as `test_longcat_lora_inference.py`) to load:
   - Native model WITHOUT LoRA
   - Native model WITH LoRA
3. Runs forward passes with identical inputs
4. Captures intermediate activations
5. **Analyzes** results by computing MSE between all combinations

## Understanding the Results

### Expected Results (if everything is correct)

```
[1] Comparing BASE models (no LoRA):
  Final output MSE: < 1e-6
  ✅ Base models match!

[2] Comparing models WITH LoRA:
  Final output MSE: < 1e-4
  ✅ LoRA models match!

[3] Effect of LoRA on original model:
  Original (no LoRA vs with LoRA) MSE: ~1e-2 to 1e-1
  LoRA changes the output significantly: True

[4] Effect of LoRA on native model:
  Native (no LoRA vs with LoRA) MSE: ~1e-2 to 1e-1
  LoRA changes the output significantly: True
```

### Problematic Results (if there's a LoRA issue)

```
[1] Comparing BASE models (no LoRA):
  Final output MSE: < 1e-6
  ✅ Base models match!

[2] Comparing models WITH LoRA:
  Final output MSE: > 1e-2
  ❌ LoRA models differ significantly  ⚠️ THIS IS THE PROBLEM

[3] Effect of LoRA on original model:
  Original (no LoRA vs with LoRA) MSE: ~1e-2
  LoRA changes the output significantly: True

[4] Effect of LoRA on native model:
  Native (no LoRA vs with LoRA) MSE: ~1e-5
  LoRA changes the output significantly: False  ⚠️ LoRA NOT WORKING
```

## How to Investigate Further

If you find divergence, you can:

1. **Check layer-by-layer activations**
   - The saved `.pt` files contain activations for blocks [0, 6, 12, 18, 24, 30, 36, 42, 47]
   - Load them and compute MSE for each layer to find where divergence starts

2. **Inspect LoRA weights**
   - Original: `/mnt/fast-disks/hao_lab/shao/LongCat-Video/weights/LongCat-Video/lora/cfg_step_lora.safetensors`
   - Native: `weights/longcat-native/lora/distilled/adapter_model.safetensors`
   - Check if keys match, shapes match, and values are similar

3. **Check LoRA application**
   - Look at `fastvideo/pipelines/lora_pipeline.py` - `set_lora_adapter()` method
   - Check if the LoRA scale is being applied correctly
   - Verify the LoRA weights are being merged into the correct layers

4. **Check conversion script**
   - `scripts/checkpoint_conversion/longcat_to_fastvideo.py`
   - Verify LoRA weights were converted correctly

## Technical Details

### Model Architecture
- 48 transformer blocks
- Each block has: self-attention, cross-attention, FFN
- LoRA is typically applied to Q, K, V projections in attention layers

### LoRA Loading Method
The native model uses the same LoRA loading as `test_longcat_lora_inference.py`:
```python
generator = VideoGenerator.from_pretrained(
    model_path,
    lora_path=lora_path,
    lora_nickname="distilled",
    ...
)
```

This ensures we're testing the actual inference path, not a manual LoRA application.

### Captured Layers
- `patch_embed` - Patch embedding
- `time_embed` - Timestep embedding
- `caption_embed` - Caption embedding
- `block_{i}_self_attn` - Self-attention output
- `block_{i}_cross_attn` - Cross-attention output
- `block_{i}_ffn` - FFN output
- `block_{i}` - Full block output
- `final_layer` - Final projection
- `final_output` - Model output

## Troubleshooting

### Error: "test_inputs.pt not found"
Run stage 1 first to create the test inputs.

### Error: Module not found
Make sure you're using the correct conda environment for each stage.

### Out of memory
The script uses small test inputs (9 frames, 30x52 spatial) to avoid OOM. If still OOM, reduce batch size or frames further.

### Models don't match even without LoRA
This suggests a problem with the base weight conversion, not LoRA. Check the conversion script.

