#!/usr/bin/env python3
"""
End-to-end image comparison (SSIM / PSNR) for Flux2 Klein across FastVideo, diffusers, and optionally SGLang.

Uses the same prompt and seed for all backends, then compares the first frame (or single image) with SSIM and PSNR.
FastVideo attention is forced to torch SDPA for reproducibility; diffusers uses its default (Flux2 transformer is not compatible with generic AttnProcessor2_0).

Usage:
  # FastVideo + diffusers only (SGLang skipped)
  python compare_flux2_e2e_ssim.py

  # Include SGLang by providing an image generated separately with the same prompt/seed
  python compare_flux2_e2e_ssim.py --sglang-image path/to/sglang_frame0.png

  # Custom prompt/seed/output dir
  python compare_flux2_e2e_ssim.py --prompt "a blue car" --seed 123 --output-dir ./my_compare

  # Run FastVideo with diffusers text encoder embeddings (to see how much TE explains the SSIM gap)
  python compare_flux2_e2e_ssim.py --fastvideo-diffusers-te

Requires: FastVideo, diffusers, torch, torchvision. For SSIM: pytorch-msssim (pip install pytorch-msssim).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch
from torchvision.io import read_video

PROMPT = "a red apple on a table"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
SEED = 42
SIZE = (1024, 1024)


def _sanitize_filename_component(name: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]', '', name)
    sanitized = sanitized.strip().strip('.')
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized or "video"


def get_fastvideo_image(
    prompt: str,
    seed: int,
    model_id: str,
    output_dir: str,
    num_gpus: int = 1,
) -> np.ndarray:
    """Generate one image with FastVideo (Flux2 Klein), return frame 0 as RGB numpy (H, W, 3) in [0, 255]."""
    from fastvideo import VideoGenerator
    from fastvideo.configs.sample import SamplingParam
    from fastvideo.attention.selector import global_force_attn_backend
    from fastvideo.platforms import AttentionBackendEnum

    # Enforce torch SDPA for fair comparison with diffusers
    global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
    os.makedirs(output_dir, exist_ok=True)
    print("Loading FastVideo generator (attention=TORCH_SDPA) ...")
    generator = VideoGenerator.from_pretrained(model_id, num_gpus=num_gpus)
    sampling = SamplingParam.from_pretrained(model_id)
    print(f"Generating (prompt={prompt!r}, seed={seed}) ...")
    generator.generate_video(
        prompt,
        sampling_param=sampling,
        output_path=output_dir,
        save_video=True,
        seed=seed,
    )
    # Output file: output_dir / "{sanitized_prompt}.mp4"
    prompt_part = _sanitize_filename_component(prompt[:100])
    mp4_path = os.path.join(output_dir, f"{prompt_part}.mp4")
    if not os.path.isfile(mp4_path):
        for f in os.listdir(output_dir):
            if f.endswith(".mp4"):
                mp4_path = os.path.join(output_dir, f)
                break
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"Expected FastVideo output at {mp4_path}")
    frames, _, _ = read_video(mp4_path, pts_unit="sec", output_format="TCHW")
    if frames.shape[0] == 0:
        raise RuntimeError(f"No frames in {mp4_path}")
    frame0 = frames[0]
    img = frame0.permute(1, 2, 0).numpy()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.max() <= 1.0:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)
    return _resize_to(img, SIZE)


def get_diffusers_prompt_embeds(
    prompt: str,
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Get Flux2 Klein prompt_embeds from diffusers (layers 9, 18, 27). Returns tensor on device."""
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        from diffusers.pipelines.flux2 import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    )
    return prompt_embeds


def get_fastvideo_image_with_diffusers_embeds(
    prompt: str,
    seed: int,
    model_id: str,
    output_dir: str,
    diffusers_prompt_embeds: torch.Tensor,
    num_gpus: int = 1,
) -> np.ndarray:
    """Run FastVideo (Flux2 Klein) with precomputed diffusers prompt_embeds; return frame 0 as RGB numpy."""
    from fastvideo import VideoGenerator
    from fastvideo.configs.sample import SamplingParam
    from fastvideo.attention.selector import global_force_attn_backend
    from fastvideo.platforms import AttentionBackendEnum

    global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
    os.makedirs(output_dir, exist_ok=True)
    # Use a subdir so we don't overwrite the normal FastVideo mp4
    fv_te_dir = os.path.join(output_dir, "fastvideo_diffusers_te")
    os.makedirs(fv_te_dir, exist_ok=True)
    print("Loading FastVideo generator (attention=TORCH_SDPA) for diffusers-TE run ...")
    generator = VideoGenerator.from_pretrained(model_id, num_gpus=num_gpus)
    sampling = SamplingParam.from_pretrained(model_id)
    print(f"Generating with diffusers prompt_embeds (prompt={prompt!r}, seed={seed}) ...")
    generator.generate_video(
        prompt,
        sampling_param=sampling,
        output_path=fv_te_dir,
        save_video=True,
        seed=seed,
        prompt_embeds=[diffusers_prompt_embeds],
    )
    prompt_part = _sanitize_filename_component(prompt[:100])
    mp4_path = os.path.join(fv_te_dir, f"{prompt_part}.mp4")
    if not os.path.isfile(mp4_path):
        for f in os.listdir(fv_te_dir):
            if f.endswith(".mp4"):
                mp4_path = os.path.join(fv_te_dir, f)
                break
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"Expected FastVideo output at {mp4_path}")
    frames, _, _ = read_video(mp4_path, pts_unit="sec", output_format="TCHW")
    if frames.shape[0] == 0:
        raise RuntimeError(f"No frames in {mp4_path}")
    frame0 = frames[0]
    img = frame0.permute(1, 2, 0).numpy()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.max() <= 1.0:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)
    return _resize_to(img, SIZE)


def get_diffusers_image(
    prompt: str,
    seed: int,
    model_id: str,
    output_dir: str,
) -> np.ndarray:
    """Generate one image with diffusers Flux2KleinPipeline, return RGB numpy (H, W, 3) in [0, 255]."""
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        from diffusers.pipelines.flux2 import Flux2KleinPipeline

    device = "cuda"
    dtype = torch.bfloat16
    print("Loading diffusers Flux2KleinPipeline ...")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    # Flux2 transformer uses Flux2Attention; AttnProcessor2_0 is incompatible (expects spatial_norm).
    # Rely on diffusers default; PyTorch 2 often uses SDPA by default.
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Generating (prompt={prompt!r}, seed={seed}) ...")
    image = pipe(
        prompt=prompt,
        height=SIZE[0],
        width=SIZE[1],
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=generator,
    ).images[0]
    img = np.array(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    out_path = os.path.join(output_dir, "diffusers.png")
    image.save(out_path)
    print(f"Saved {out_path}")
    return _resize_to(img, SIZE)


def load_image(path: str) -> np.ndarray:
    """Load image from path to RGB numpy (H, W, 3) in [0, 255]."""
    from PIL import Image
    img = np.array(Image.open(path).convert("RGB"))
    return _resize_to(img, SIZE)


def _resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (height, width) if needed; return uint8."""
    h, w = size
    if img.shape[0] == h and img.shape[1] == w:
        return img
    from PIL import Image
    pil = Image.fromarray(img)
    pil = pil.resize((w, h), Image.Resampling.LANCZOS)
    return np.array(pil)


def compute_ssim_psnr(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """Compute SSIM and PSNR between two RGB images (H, W, 3) in [0, 255]. Returns (ssim, psnr)."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    a = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    b = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    try:
        from pytorch_msssim import ssim as torch_ssim
        ssim_val = torch_ssim(a, b, data_range=1.0).item()
    except ImportError:
        try:
            from skimage.metrics import structural_similarity as ssim_sk
            ssim_val = ssim_sk(img1, img2, channel_axis=2, data_range=255)
        except ImportError:
            ssim_val = float("nan")
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    psnr_val = 10 * np.log10(255.0 ** 2 / (mse + 1e-10))
    return ssim_val, psnr_val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E image comparison (SSIM/PSNR) for Flux2 Klein: FastVideo, diffusers, SGLang."
    )
    parser.add_argument("--prompt", default=PROMPT, help="Text prompt")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (positive integer)")
    parser.add_argument("--model-id", default=MODEL_ID, help="Model ID")
    parser.add_argument("--output-dir", default="./e2e_compare", help="Directory for outputs and saved images")
    parser.add_argument("--sglang-image", default=None, help="Path to image from SGLang (same prompt/seed) to include")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for FastVideo")
    parser.add_argument("--skip-fastvideo", action="store_true", help="Skip FastVideo generation")
    parser.add_argument("--skip-diffusers", action="store_true", help="Skip diffusers generation")
    parser.add_argument(
        "--fastvideo-diffusers-te",
        action="store_true",
        help="Also run FastVideo with diffusers text encoder embeddings to measure how much TE explains SSIM gap",
    )
    args = parser.parse_args()

    if args.seed < 1:
        print("Warning: seed must be positive for FastVideo; using 1 instead of", args.seed)
        args.seed = 1

    os.makedirs(args.output_dir, exist_ok=True)
    images: dict[str, np.ndarray] = {}

    if not args.skip_fastvideo:
        try:
            images["FastVideo"] = get_fastvideo_image(
                args.prompt, args.seed, args.model_id, args.output_dir, num_gpus=args.num_gpus
            )
        except Exception as e:
            print(f"FastVideo failed: {e}")
            raise
    else:
        prompt_part = _sanitize_filename_component(args.prompt[:100])
        mp4_path = os.path.join(args.output_dir, f"{prompt_part}.mp4")
        if not os.path.isfile(mp4_path):
            for f in os.listdir(args.output_dir):
                if f.endswith(".mp4"):
                    mp4_path = os.path.join(args.output_dir, f)
                    break
        if os.path.isfile(mp4_path):
            frames, _, _ = read_video(mp4_path, pts_unit="sec", output_format="TCHW")
            frame0 = frames[0].permute(1, 2, 0).numpy()
            if frame0.max() <= 1.0:
                frame0 = (frame0 * 255).clip(0, 255).astype(np.uint8)
            images["FastVideo"] = _resize_to(frame0, SIZE)
        else:
            print("--skip-fastvideo but no mp4 found in output-dir; skipping FastVideo")

    if not args.skip_diffusers:
        images["diffusers"] = get_diffusers_image(
            args.prompt, args.seed, args.model_id, args.output_dir
        )

    if args.fastvideo_diffusers_te:
        if "diffusers" not in images:
            print("--fastvideo-diffusers-te requires diffusers; running diffusers first.")
            images["diffusers"] = get_diffusers_image(
                args.prompt, args.seed, args.model_id, args.output_dir
            )
        print("Getting diffusers prompt_embeds ...")
        dtype = torch.bfloat16
        diffusers_embeds = get_diffusers_prompt_embeds(
            args.prompt, args.model_id, device="cuda", dtype=dtype
        )
        images["FastVideo (diffusers TE)"] = get_fastvideo_image_with_diffusers_embeds(
            args.prompt,
            args.seed,
            args.model_id,
            args.output_dir,
            diffusers_embeds,
            num_gpus=args.num_gpus,
        )

    if args.sglang_image and os.path.isfile(args.sglang_image):
        images["SGLang"] = load_image(args.sglang_image)
        print(f"Loaded SGLang image from {args.sglang_image}")
    elif args.sglang_image:
        print(f"Warning: --sglang-image {args.sglang_image} not found; skipping SGLang")

    if len(images) < 2:
        print("Need at least two images to compare. Run without --skip-* and/or pass --sglang-image.")
        sys.exit(1)

    for name, img in images.items():
        out = os.path.join(args.output_dir, f"compare_{name.lower().replace(' ', '_')}.png")
        from PIL import Image
        Image.fromarray(img).save(out)
        print(f"Saved {out}")

    names = list(images.keys())
    print("\n--- SSIM / PSNR (same prompt & seed) ---")
    print(f"  Prompt: {args.prompt!r}")
    print(f"  Seed:   {args.seed}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ssim_val, psnr_val = compute_ssim_psnr(images[a], images[b])
            print(f"  {a} vs {b}:  SSIM = {ssim_val:.4f}   PSNR = {psnr_val:.2f} dB")
    print("Done.")


if __name__ == "__main__":
    main()
