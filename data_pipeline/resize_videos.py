# SPDX-License-Identifier: Apache-2.0
"""Stage 2 (VAE-free variant): crop+resize source videos to the training geometry.

Applies exactly the same ``center_crop_th_tw`` + ``resize`` transform as
``decode_roundtrip_videos.py`` (and as Stage 5's ``CenterCropResizeVideo``), but skips
the VAE encode/decode. Tracks extracted from these videos land in the same coordinate
frame as the training latents; they just don't carry the VAE's reconstruction artifacts.

Purpose: the geometry is what track alignment *requires*; the VAE round-trip is what it
*may* require. This script exists so the two can be A/B'd -- extract tracks from
``resized_videos/`` and from ``roundtrip_videos/``, then diff the npz. If the track delta
is small, large-scale runs can skip the VAE pass entirely (it is pure GPU cost per clip).

Usage:
    python data_pipeline/resize_videos.py \\
        --data-dir /home/hal-kevin/data/motion-stream-test

CPU-only; parallelize with --index / --limit sharding if needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastvideo.dataset.transform import center_crop_th_tw, resize

TARGET_H, TARGET_W = 480, 832
NUM_FRAMES = 121


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="Dataset root (contains videos/, etc.).")
    p.add_argument("--video-subdir", type=str, default="videos", help="Input video subdirectory.")
    p.add_argument("--out-subdir", type=str, default="resized_videos", help="Output subdirectory.")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    p.add_argument("--height", type=int, default=TARGET_H)
    p.add_argument("--width", type=int, default=TARGET_W)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--index", type=int, nargs="+", default=None, metavar="IDX",
                   help="Process only these video indices (e.g. --index 4 7 12). Assumes vid_%06d naming.")
    p.add_argument("--include-list", type=Path, default=None,
                   help="Text file of clip filenames (one per line) to process; everything else in "
                        "--video-subdir is ignored. Pairs with filter_clips.py's needs_resize.txt "
                        "so only off-spec clips are re-encoded.")
    p.add_argument("--limit", type=int, default=None, help="Process only first N videos (smoke test).")
    p.add_argument("--rank", type=int, default=0, help="Shard index for CPU-parallel runs (0-indexed).")
    p.add_argument("--world-size", type=int, default=1, help="Total number of parallel processes.")
    p.add_argument("--min-frames", type=int, default=None,
                   help="Skip clips with fewer than this many frames (default: --num-frames). "
                        "Set 0 to keep short clips (output T then varies per clip).")
    p.add_argument("--force", action="store_true", help="Re-write even if output already exists.")
    return p.parse_args()


def resize_video(path: Path, num_frames: int, height: int, width: int) -> np.ndarray:
    """Crop+resize to the training geometry. Returns uint8 frames [T, H, W, C].

    Reads sequentially and stops at num_frames or end-of-file, so clips shorter than
    num_frames yield what they have rather than raising (real-world shards are ragged).
    """
    reader = imageio.get_reader(str(path))
    frames = []
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
        frames.append(np.asarray(frame))
    reader.close()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    clip = center_crop_th_tw(clip, height, width, top_crop=False)
    clip = resize(clip, (height, width), interpolation_mode="bilinear")
    return (clip.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()


def main() -> None:
    args = parse_args()
    videos_dir = args.data_dir / args.video_subdir
    out_dir = args.data_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob("*.mp4"))
    if args.include_list is not None:
        wanted = {ln.strip() for ln in args.include_list.read_text().splitlines() if ln.strip()}
        videos = [v for v in videos if v.name in wanted]
    if args.index is not None:
        wanted = {f"vid_{i:06d}.mp4" for i in args.index}
        videos = [v for v in videos if v.name in wanted]
    if args.limit is not None:
        videos = videos[:args.limit]
    if args.world_size > 1:
        videos = videos[args.rank::args.world_size]
    if not videos:
        print(f"[resize] no videos found in {videos_dir}", flush=True)
        return

    min_frames = args.num_frames if args.min_frames is None else args.min_frames
    print(f"[resize] {len(videos)} videos → {out_dir} ({args.height}x{args.width}, no VAE)"
          f"{f' [shard {args.rank}/{args.world_size}]' if args.world_size > 1 else ''}", flush=True)
    n_ok = n_short = n_err = 0
    for k, vpath in enumerate(videos, 1):
        out_path = out_dir / vpath.name
        if out_path.exists() and not args.force:
            continue
        try:
            frames = resize_video(vpath, args.num_frames, args.height, args.width)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[resize] [{k}/{len(videos)}] {vpath.name}: DECODE FAILED ({e}), skipping", flush=True)
            continue
        if frames.shape[0] < min_frames:
            n_short += 1
            print(f"[resize] [{k}/{len(videos)}] {vpath.name}: only {frames.shape[0]} frames "
                  f"(< {min_frames}), skipping", flush=True)
            continue
        # Dot-prefixed so a leftover temp is NOT picked up by downstream `*.mp4` globs
        # (a stale "<name>.tmp.mp4" once got fed to the tracker and killed the worker).
        tmp = out_path.with_name(f".{out_path.stem}.tmp.mp4")
        imageio.mimsave(str(tmp), frames, fps=args.fps, macro_block_size=1)
        tmp.replace(out_path)
        n_ok += 1
        if k % 50 == 0 or k == len(videos):
            print(f"[resize] [{k}/{len(videos)}] ok={n_ok} short={n_short} err={n_err}", flush=True)

    print(f"[resize] done. ok={n_ok} short={n_short} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
