# LoRA Extraction and Merging

Tools for extracting and merging LoRA adapters for FastVideo models.

## Extract LoRA Adapter

```bash
python scripts/lora_extraction/extract_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --out adapter_r32.safetensors \
  --rank 32
```

**Options:**
- `--base`: Base model (HuggingFace ID or local path)
- `--ft`: Fine-tuned model (HuggingFace ID or local path)
- `--out`: Output adapter file
- `--rank`: LoRA rank (16, 32, 64, 128)
- `--full-rank`: Extract full-rank adapter (optional)

## Merge Adapter

```bash
python scripts/lora_extraction/merge_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --adapter adapter_r32.safetensors \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --output merged_model
```

**Options:**
- `--base`: Base model (HuggingFace ID or local path)
- `--adapter`: LoRA adapter file (.safetensors)
- `--ft`: Fine-tuned model (for configuration)
- `--output`: Output directory

## Validate Quality (Optional)

```bash
python scripts/lora_extraction/lora_inference_comparison.py \
  --base merged_model \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter NONE \
  --output-dir results \
  --prompt "A cat sitting on a windowsill" \
  --seed 42 \
  --height 480 \
  --width 480 \
  --num-frames 49 \
  --num-inference-steps 32 \
  --compute-ssim \
  --compute-lpips
```

**Options:**
- `--base`: Merged model or base model path
- `--ft`: Fine-tuned model (reference)
- `--adapter`: Path to adapter or NONE
- `--output-dir`: Output directory
- `--prompt`: Text prompt (default: "A cat sitting on a windowsill")
- `--seed`: Random seed (default: 42)
- `--height`: Video height (default: 480)
- `--width`: Video width (default: 832)
- `--num-frames`: Number of frames (default: 49)
- `--num-inference-steps`: Inference steps (default: 32)
- `--compute-ssim`: Compute SSIM metric
- `--compute-lpips`: Compute LPIPS metric
