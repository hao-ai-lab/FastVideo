# LoRA Extraction and Verification Scripts

Utility scripts for extracting, verifying, merging and comparing LoRA adapters between a base diffusion model and a fine-tuned FastVideo model.

Intended models: Wan 2.2 TI2V style models (e.g. Wan-AI/Wan2.2-TI2V-5B-Diffusers and FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers).

---

## Scripts Overview

Contents (scripts)

extract_lora.py — Extract LoRA adapters by SVD of (fine_tuned_weight - base_weight) and save adapters (safetensors preferred). Supports checkpointing and resume.

verify_lora.py — Reconstruct base + (B @ A) from the adapter and compare to the fine-tuned model (numeric verification on sampled layers).

merge_lora.py — Merge adapter into base transformer weights offline and write merged transformer weights (HF diffusers / safetensors friendly).

lora_inference_comparison.py — Generate videos with the fine-tuned model and with merged/base+adapter using the same seed; optionally compute SSIM to quantify similarity.

---

## Usage

Usage examples

Run commands from the repository root (so fastvideo can be imported)

Run extraction:
```bash
python extract_lora.py --base <base> --ft <ft> --out adapter.safetensors --rank 16
```

Verify:
```bash
python verify_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter adapter.safetensors \
  --samples 100

```

Merge Adapter into base model:
```base
python scripts/lora_extraction/merge_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --adapter artifacts/fastvideo_adapter.r16.safetensors \
  --output ./merged_model_base_lora
```

Compare inference outputs:

with offline merge:
```bash
python scripts/lora_extraction/lora_inference_comparison.py \
  --base ./merged_model \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter NONE \
  --output-dir ./inference_comparison \
  --compute-ssim \
  --seed 42
```

with online merge:
```bash
python scripts/lora_extraction/lora_inference_comparison.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter adapter.safetensors \
  --output-dir ./inference_comparison \
  --compute-ssim \
  --seed 42

```
