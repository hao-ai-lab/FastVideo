# SPDX-License-Identifier: Apache-2.0
"""Stage 2: produce VAE round-trip videos for CoTracker.

Encode each source video through the FastVideo WanVAE (use_feature_cache=False,
with causal-boundary fix) then immediately decode it back. The resulting videos
differ slightly from the originals (compression artifacts, mild color shift) but
are exactly what Stage 5 (preprocess_to_parquet) will store as latents and what
the validation callback will decode for reference. CoTracker tracks extracted from
these round-trip videos therefore align with training latents and the validation
reference display.

Usage:
    python data_pipeline/decode_roundtrip_videos.py \\
        --data-dir /home/hal-kevin/data/motion-physics \\
        --vae-path /home/hal-kevin/models/trackwan_1.3b_i2v_control_init/vae

    # Re-run specific indices only
    python data_pipeline/decode_roundtrip_videos.py \\
        --data-dir /home/hal-kevin/data/motion-physics \\
        --vae-path /home/hal-kevin/models/trackwan_1.3b_i2v_control_init/vae \\
        --index 4 7 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastvideo.configs.models.vaes.wanvae import WanVAEConfig
from fastvideo.dataset.transform import center_crop_th_tw, resize
from fastvideo.models.vaes.wanvae import AutoencoderKLWan

TARGET_H, TARGET_W = 480, 832
NUM_FRAMES = 121


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="Dataset root (contains videos/, etc.).")
    p.add_argument("--vae-path", type=Path, required=True, help="Path to the VAE directory (contains diffusion_pytorch_model.safetensors).")
    p.add_argument("--video-subdir", type=str, default="videos", help="Input video subdirectory.")
    p.add_argument("--out-subdir", type=str, default="roundtrip_videos", help="Output subdirectory.")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="Number of frames per video.")
    p.add_argument("--height", type=int, default=TARGET_H)
    p.add_argument("--width", type=int, default=TARGET_W)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--index", type=int, nargs="+", default=None, metavar="IDX",
                   help="Process only these video indices (e.g. --index 4 7 12).")
    p.add_argument("--limit", type=int, default=None, help="Process only first N videos (smoke test).")
    p.add_argument("--force", action="store_true", help="Re-encode even if output already exists.")
    return p.parse_args()


def load_vae(vae_path: Path, device: str) -> AutoencoderKLWan:
    config = WanVAEConfig(use_feature_cache=False)
    vae = AutoencoderKLWan(config).to(device).eval()
    weights = safetensors_load_file(str(vae_path / "diffusion_pytorch_model.safetensors"))
    vae.load_state_dict(weights, strict=True)
    return vae


def load_video(path: Path, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Return pixel tensor [1, C, T, H, W] in [-1, 1]."""
    reader = imageio.get_reader(str(path))
    frames = [reader.get_data(i) for i in range(num_frames)]
    reader.close()
    clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    clip = center_crop_th_tw(clip, height, width, top_crop=False)
    clip = resize(clip, (height, width), interpolation_mode="bilinear")
    pixel = (clip * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,T,H,W]
    return pixel


@torch.no_grad()
def roundtrip(vae: AutoencoderKLWan, pixel: torch.Tensor, device: str) -> np.ndarray:
    """Return decoded frames as uint8 numpy [T, H, W, C]."""
    pixel = pixel.to(device)
    with torch.autocast(device, dtype=torch.float32):
        latent = vae.encode(pixel).mean
    decoded = vae.decode(latent)
    frames = decoded[0].permute(1, 2, 3, 0).float().cpu()
    frames = ((frames / 2 + 0.5).clamp(0, 1) * 255).byte().numpy()
    return frames


def main() -> None:
    args = parse_args()
    device = args.device

    videos_dir = args.data_dir / args.video_subdir
    out_dir = args.data_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob("*.mp4"))
    if args.index is not None:
        wanted = {f"vid_{i:06d}.mp4" for i in args.index}
        videos = [v for v in videos if v.name in wanted]
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        print(f"[roundtrip] no videos found in {videos_dir}", flush=True)
        return

    print(f"[roundtrip] loading VAE from {args.vae_path} ...", flush=True)
    vae = load_vae(args.vae_path, device)

    print(f"[roundtrip] {len(videos)} videos → {out_dir}", flush=True)
    for k, vpath in enumerate(videos, 1):
        out_path = out_dir / vpath.name
        if out_path.exists() and not args.force:
            print(f"[roundtrip] [{k}/{len(videos)}] {vpath.name} already exists, skipping", flush=True)
            continue

        pixel = load_video(vpath, args.num_frames, args.height, args.width)
        frames = roundtrip(vae, pixel, device)
        imageio.mimsave(str(out_path), frames, fps=args.fps, macro_block_size=1)
        print(f"[roundtrip] [{k}/{len(videos)}] {vpath.name} → {out_path.name}  "
              f"shape={frames.shape}", flush=True)

    print("[roundtrip] done.", flush=True)


if __name__ == "__main__":
    main()
