"""Decode 'real' latents from a KD cache to videos.

Uses fastvideo's AutoencoderKLWan and its built-in denormalization
(latent * std + mean, matching normalize_dit_input's inverse).

Usage:
    python examples/train/scenario/ode_init_self_forcing_wan_causal/decode_kd_cache.py \
        --cache_dir data/kd_test_cache_small \
        --out_dir   data/kd_test_cache_small/video \
        --fps 16
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def denormalize(latents: torch.Tensor, vae: AutoencoderKLWan) -> torch.Tensor:
    """Inverse of normalize_dit_input('wan', ...).

    normalize_dit_input: normalized = (latent - mean) * (1/std)
    inverse:             latent     = normalized * std + mean
    """
    cfg = vae.config
    mean = torch.tensor(cfg.latents_mean, dtype=latents.dtype,
                        device=latents.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(cfg.latents_std, dtype=latents.dtype,
                       device=latents.device).view(1, -1, 1, 1, 1)
    return latents * std + mean


def decode_real(pt_path: Path, vae: AutoencoderKLWan,
                device: torch.device) -> np.ndarray:
    """Load one .pt cache file and decode its 'real' latent to uint8 [T,H,W,3]."""
    d = torch.load(pt_path, weights_only=True, map_location="cpu")
    real = d["real"].float()  # [T, C, H, W]  (normalized)

    # [T, C, H, W] → [1, C, T, H, W]
    latent = real.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    latent = denormalize(latent, vae)

    with torch.no_grad():
        video = vae.decode(latent).sample  # [1, 3, T_px, H_px, W_px]

    video = video.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, 3]
    video = ((video.clamp(-1, 1) + 1) / 2 * 255).cpu().float().numpy()
    return video.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data/kd_test_cache_small")
    parser.add_argument("--out_dir", default="data/kd_test_cache_small/video")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    samples_dir = Path(args.cache_dir) / "samples"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VAE from {args.model_id} ...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32)
    vae.eval().to(device)

    pts = sorted(samples_dir.glob("*.pt"))
    print(f"Found {len(pts)} samples in {samples_dir}")

    for pt in pts:
        out_path = out_dir / (pt.stem + ".mp4")
        if out_path.exists() and not args.overwrite:
            print(f"  skip {pt.name} (exists)")
            continue
        video_np = decode_real(pt, vae, device)
        imageio.mimsave(str(out_path), video_np, fps=args.fps)
        print(f"  {pt.name} → {out_path.name}  {video_np.shape}")

    print(f"Done. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
