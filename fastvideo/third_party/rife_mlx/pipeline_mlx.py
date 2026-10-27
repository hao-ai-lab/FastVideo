"""High-level interpolation + CLI (image-pair + full video). P5.

    rife-mlx -i in.mp4 -o out.mp4 --multi 2 --scale 1.0
    rife-mlx --img0 a.png --img1 b.png -t 0.5 -o mid.png

--multi N inserts N-1 frames between each pair (at t = i/N); output fps = N*input.
--scale {0.25,0.5,1,2,4} sets the coarse-to-fine pyramid scale (0.5 for 4K).
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np

from .config import DEFAULT_VERSION
from .utils.weights import build_model


def _to_nhwc(img_u8: np.ndarray) -> mx.array:
    return mx.array(img_u8.astype(np.float32)[None] / 255.0)


def _to_u8(arr: mx.array) -> np.ndarray:
    a = np.clip(np.array(arr)[0], 0.0, 1.0)
    return (a * 255.0).round().astype(np.uint8)


def interpolate_pair(model, img0_u8, img1_u8, timestep=0.5, scale=1.0) -> np.ndarray:
    out = model.inference(_to_nhwc(img0_u8), _to_nhwc(img1_u8), timestep, scale)
    mx.eval(out)
    return _to_u8(out)


def interpolate_sequence(model, frames, multi=2, scale=1.0):
    """frames: list[HWC uint8] -> list with multi-1 frames inserted per pair."""
    out = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        for j in range(1, multi):
            out.append(interpolate_pair(model, frames[i], frames[i + 1], j / multi, scale))
    out.append(frames[-1])
    return out


def cli_main() -> None:
    p = argparse.ArgumentParser(description="Practical-RIFE 4.25 MLX frame interpolation")
    p.add_argument("-i", "--input", help="input video")
    p.add_argument("--img0"); p.add_argument("--img1")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-t", "--timestep", type=float, default=0.5, help="image-pair mode")
    p.add_argument("-m", "--multi", type=int, default=2, help="video fps multiplier")
    p.add_argument("-s", "--scale", type=float, default=1.0,
                   choices=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("-n", "--version", default=DEFAULT_VERSION)
    p.add_argument("--weights_dir", default=None)
    args = p.parse_args()

    model = build_model(args.version, weights_dir=args.weights_dir)

    if args.img0 and args.img1:
        from PIL import Image
        a = np.asarray(Image.open(args.img0).convert("RGB"))
        b = np.asarray(Image.open(args.img1).convert("RGB"))
        mid = interpolate_pair(model, a, b, args.timestep, args.scale)
        Image.fromarray(mid).save(args.output)
        print(f"{args.img0} + {args.img1} @t={args.timestep} -> {args.output} {mid.shape}")
        return

    from .video import read_frames, write_video
    frames, fps = read_frames(args.input)
    out_frames = interpolate_sequence(model, frames, args.multi, args.scale)
    write_video(args.output, out_frames, fps * args.multi, audio_from=args.input)
    print(f"{args.input} ({len(frames)}f @{fps:.2f}) -> {args.output} "
          f"({len(out_frames)}f @{fps * args.multi:.2f}, audio passthrough)")


if __name__ == "__main__":
    cli_main()
