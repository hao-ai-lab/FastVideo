"""
Compare fine-tuned FastVideo outputs vs. base + extracted LoRA applied to base transformer.

Usage:
    python compare_lora_outputs.py

Requirements:
    pip install diffusers transformers torch pillow tqdm lpips scikit-image

Notes:
 - Runs both FT and base+LoRA pipelines for the same prompts and seeds.
 - Computes MSE, MAE, optional LPIPS and SSIM.
 - Saves outputs + metrics to batch_compare_out/.
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import DiffusionPipeline

# CONFIG
BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FINETUNED_MODEL = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
LORA_PATH = "fastwan2.2_transformer_lora.pt"
OUTPUT_DIR = "batch_compare_out"

PROMPTS = [
    "A cinematic futuristic city at sunset, ultra-detailed",
    "A lone astronaut walking through a dense jungle, photorealistic",
    "Macro shot of a dewdrop on a leaf, shallow depth of field",
    "A medieval town market at dawn, painterly style",
    "A portrait of an elderly woman under Rembrandt lighting"
]

NUM_STEPS = 12
SEED_BASE = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional metrics
compute_lpips = False
compute_ssim = False
lpips_model = None
try:
    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
    compute_lpips = True
except Exception:
    pass
try:
    from skimage.metrics import structural_similarity as ssim_fn
    compute_ssim = True
except Exception:
    pass

# Utility functions
def to_uint8(arr):
    """Normalize tensor/array to uint8 [0,255]."""
    arr = np.asarray(arr)
    if arr.ndim == 4:  # (B,C,H,W) or (B,H,W,C)
        arr = arr[0]
        if arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
            arr = arr.transpose(1,2,0)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() < -0.5:
            arr = (arr + 1.0) / 2.0
        arr = (arr * 255).clip(0,255).astype(np.uint8)
    return arr

def tensor_for_lpips(pil_img):
    """Convert PIL image to LPIPS-compatible tensor [-1,1], NCHW."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = arr.transpose(2,0,1)
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

def extract_pil_from_output(output):
    """Extract PIL image from Diffusers output."""
    if hasattr(output, "images"):
        return output.images[0]
    if hasattr(output, "image"):
        return output.image[0]
    if isinstance(output, Image.Image):
        return output
    if torch.is_tensor(output):
        return Image.fromarray(to_uint8(output.cpu().numpy()))
    raise ValueError("Cannot extract image from output.")

def run_pipeline(pipe, prompts, tag):
    """Generate images for given prompts and save to disk."""
    paths = []
    for i, prompt in enumerate(tqdm(prompts, desc=f"{tag} generation")):
        seed = SEED_BASE + i
        gen = torch.manual_seed(seed)
        out = pipe(prompt, num_inference_steps=NUM_STEPS, generator=gen)
        img = extract_pil_from_output(out)
        path = os.path.join(OUTPUT_DIR, f"{tag}_{i}.png")
        img.save(path)
        paths.append(path)
    return paths

# Main logic
def main():
    print(f"Using device: {DEVICE}")

    print("\nLoading base model (CPU)")
    base_pipe = DiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    base_sd = base_pipe.transformer.state_dict()
    del base_pipe
    torch.cuda.empty_cache()

    print("Applying LoRA deltas to base weights")
    lora_dict = torch.load(LORA_PATH, map_location="cpu")
    for key, val in lora_dict.items():
        if key in base_sd:
            A, B = val["A"].float(), val["B"].float()
            base_sd[key] = base_sd[key].float() + (A @ B.T)
    print(f"LoRA applied to {len(lora_dict)} keys.\n")

    print("Running fine-tuned pipeline (reference)")
    ft_pipe = DiffusionPipeline.from_pretrained(FINETUNED_MODEL, torch_dtype=DTYPE).to(DEVICE)
    ref_images = run_pipeline(ft_pipe, PROMPTS, "ref")
    del ft_pipe
    torch.cuda.empty_cache()

    print("\nRunning base+LoRA pipeline")
    mod_pipe = DiffusionPipeline.from_pretrained(FINETUNED_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    mod_pipe.transformer.load_state_dict(base_sd, strict=False)
    mod_pipe.to(DEVICE)
    mod_images = run_pipeline(mod_pipe, PROMPTS, "mod")
    del mod_pipe
    torch.cuda.empty_cache()

    print("\nComputing metrics")
    results = []
    for i, prompt in enumerate(PROMPTS):
        ref = np.array(Image.open(ref_images[i])).astype(np.float32) / 255.0
        mod = np.array(Image.open(mod_images[i])).astype(np.float32) / 255.0
        mse = float(np.mean((ref - mod)**2))
        mae = float(np.mean(np.abs(ref - mod)))
        lpips_val = None
        ssim_val = None

        if compute_lpips:
            with torch.no_grad():
                lpips_val = float(lpips_model(tensor_for_lpips(Image.open(ref_images[i])),
                                              tensor_for_lpips(Image.open(mod_images[i]))).item())
        if compute_ssim:
            try:
                ssim_val = float(ssim_fn(ref, mod, channel_axis=2, data_range=1.0))
            except Exception:
                ssim_val = None

        results.append({"i": i, "prompt": prompt, "mse": mse, "mae": mae, "lpips": lpips_val, "ssim": ssim_val})
        print(f"[{i}] MSE={mse:.3e}, MAE={mae:.3e}, LPIPS={lpips_val}, SSIM={ssim_val}")

    # Summary
    mse_vals = [r["mse"] for r in results]
    mae_vals = [r["mae"] for r in results]
    print("\nSummary:")
    print(f"MSE mean={np.mean(mse_vals):.3e}")
    print(f"MAE mean={np.mean(mae_vals):.3e}")
    if compute_lpips:
        print(f"LPIPS mean={np.mean([r['lpips'] for r in results if r['lpips']]):.3e}")
    if compute_ssim:
        print(f"SSIM mean={np.mean([r['ssim'] for r in results if r['ssim']]):.4f}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["i","prompt","mse","mae","lpips","ssim"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
