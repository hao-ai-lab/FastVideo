# SPDX-License-Identifier: Apache-2.0
"""Score a Regenerate-2K run against the base-H3 reference frames and write side-by-side crops.

The reference is ``ref_frames_u8.pt`` ([121, H, W, 3] uint8) from the parity run in
``/mnt/nfs/vlm-aryan/regen2k/out/parity_fox/``. PSNR/SSIM are computed on the luma of every sixth frame, matching
``compare_parity.py`` in that directory, plus a crop image per requested frame index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def gaussian_window(size: int = 11, sigma: float = 1.5, device: str = "cpu") -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x.float(), y.float())
    return float(10 * torch.log10(255.0**2 / mse)) if mse > 0 else float("inf")


def ssim(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor) -> float:
    x = x.float().mean(-1)[None, None]
    y = y.float().mean(-1)[None, None]
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    sxx = F.conv2d(x * x, window) - mu_x * mu_x
    syy = F.conv2d(y * y, window) - mu_y * mu_y
    sxy = F.conv2d(x * y, window) - mu_x * mu_y
    return float((((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2))).mean())


def load_frames(path: Path) -> torch.Tensor:
    frames = torch.as_tensor(torch.load(path))
    if frames.ndim == 5:
        frames = frames[0]
    if frames.shape[0] == 3:
        frames = frames.permute(1, 2, 3, 0)
    if frames.dtype != torch.uint8:
        frames = (frames.float().clamp(0, 1) * 255).round().to(torch.uint8) if frames.max() <= 1.0 else frames.round(
        ).clamp(0, 255).to(torch.uint8)
    return frames


def write_crop(reference: torch.Tensor, candidate: torch.Tensor, index: int, path: Path, size: int = 512) -> None:
    height, width = reference.shape[1:3]
    top = (height - size) // 4
    left = (width - size) // 2
    crops = [frame[index, top:top + size, left:left + size] for frame in (reference, candidate)]
    Image = __import__("PIL.Image", fromlist=["Image"])
    stacked = torch.cat(crops, dim=1).numpy()
    Image.fromarray(stacked).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="reference frames .pt")
    parser.add_argument("--dir", required=True, help="a run directory written by run_regen.py")
    parser.add_argument("--frames", default="0,60,120")
    args = parser.parse_args()
    run_dir = Path(args.dir)
    reference = load_frames(Path(args.ref))
    video = run_dir / "run_0.mp4"
    frames_path = run_dir / "frames.pt"
    if not frames_path.is_file():
        import imageio.v3 as iio

        frames = torch.from_numpy(iio.imread(video, plugin="pyav"))
        torch.save(frames, frames_path)
    candidate = load_frames(frames_path)

    count = min(len(reference), len(candidate))
    indices = list(range(0, count, 6))
    window = gaussian_window()
    scores = {
        "frames": count,
        "psnr_db": sum(psnr(candidate[i], reference[i]) for i in indices) / len(indices),
        "ssim": sum(ssim(candidate[i], reference[i], window) for i in indices) / len(indices),
    }
    write_crop(reference, candidate, 0, run_dir / "crop_000.png")
    for index in (int(part) for part in args.frames.split(",")):
        if index < count:
            write_crop(reference, candidate, index, run_dir / f"crop_{index:03d}.png")
    (run_dir / "quality.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
