# SPDX-License-Identifier: Apache-2.0
"""Visualize CoTracker point tracks over a generated video.

Produces, for one (video, tracks) pair:
- ``<out>_overlay.mp4``: every frame with visible points drawn as dots plus a short
  motion tail (last ``--tail`` frames). Colour encodes the point's initial grid position.
- ``<out>_trajectories.png``: all full trajectories drawn over frame 0 (static summary).

Pure CPU; depends only on numpy + PIL + imageio + torchvision (already in the venv), so it
does NOT need CoTracker or a GPU.

Example:
    .venv/bin/python data_pipeline/visualize_tracks.py \
        --video  /.../smoke_wan21_1.3b_480p/videos/vid_000000.mp4 \
        --tracks /.../smoke_wan21_1.3b_480p/tracks/vid_000000.npz \
        --out    /.../smoke_wan21_1.3b_480p/viz/vid_000000
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="Source .mp4")
    p.add_argument("--tracks", type=Path, required=True, help=".npz from extract_tracks.py")
    p.add_argument("--out", type=Path, required=True, help="Output path prefix (no extension).")
    p.add_argument("--stride", type=int, default=2, help="Sub-sample the NxN grid by this factor for clarity.")
    p.add_argument("--tail", type=int, default=12, help="Motion-tail length in frames.")
    p.add_argument("--radius", type=int, default=2, help="Point radius in px.")
    p.add_argument("--fps", type=int, default=16, help="Output video fps.")
    p.add_argument("--vis-thresh", type=float, default=0.5, help="Visibility threshold.")
    return p.parse_args()


def load_frames(path: Path) -> np.ndarray:
    """Return frames (T, H, W, 3) uint8.

    Recent torchvision dropped ``torchvision.io.read_video``; use decord, falling
    back to imageio's ffmpeg reader.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(path), ctx=cpu(0))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    except Exception:  # noqa: BLE001 - fall back to ffmpeg
        reader = imageio.get_reader(str(path), format="ffmpeg")
        frames = np.stack([np.asarray(f) for f in reader], axis=0)
        reader.close()
    return frames[..., :3].astype(np.uint8)


def grid_colors(grid_size: int, stride: int) -> np.ndarray:
    """One RGB colour per (sub-sampled) grid point, encoding its initial position."""
    idx = np.arange(0, grid_size, stride)
    gy, gx = np.meshgrid(idx, idx, indexing="ij")
    nx = gx.reshape(-1) / max(grid_size - 1, 1)
    ny = gy.reshape(-1) / max(grid_size - 1, 1)
    cols = np.empty((nx.shape[0], 3), dtype=np.uint8)
    for i, (x, y) in enumerate(zip(nx, ny)):
        r, g, b = colorsys.hsv_to_rgb(float(x), 1.0, 0.5 + 0.5 * float(y))
        cols[i] = (int(r * 255), int(g * 255), int(b * 255))
    return cols


def subsample(tracks: np.ndarray, vis: np.ndarray, grid_size: int, stride: int):
    """tracks (T,N,2), vis (T,N) with N==grid_size**2 -> sub-sampled by stride in both grid dims."""
    t = tracks.shape[0]
    if tracks.shape[1] != grid_size * grid_size:
        return tracks, vis  # unknown layout; keep as-is
    tr = tracks.reshape(t, grid_size, grid_size, 2)[:, ::stride, ::stride, :].reshape(t, -1, 2)
    vs = vis.reshape(t, grid_size, grid_size)[:, ::stride, ::stride].reshape(t, -1)
    return tr, vs


def draw_overlay(frames, tracks, vis, colors, tail, radius, vis_thresh) -> list[np.ndarray]:
    t, h, w, _ = frames.shape
    n = tracks.shape[1]
    out = []
    for fi in range(t):
        img = Image.fromarray(frames[fi]).convert("RGB")
        draw = ImageDraw.Draw(img)
        t0 = max(0, fi - tail)
        for pi in range(n):
            col = tuple(int(c) for c in colors[pi])
            # tail: consecutive visible positions in the window
            pts = []
            for tj in range(t0, fi + 1):
                if vis[tj, pi] >= vis_thresh:
                    x, y = float(tracks[tj, pi, 0]), float(tracks[tj, pi, 1])
                    if 0 <= x < w and 0 <= y < h:
                        pts.append((x, y))
                else:
                    pts = []  # break the tail on occlusion
            if len(pts) >= 2:
                draw.line(pts, fill=col, width=1)
            if vis[fi, pi] >= vis_thresh:
                x, y = float(tracks[fi, pi, 0]), float(tracks[fi, pi, 1])
                if 0 <= x < w and 0 <= y < h:
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=col)
        out.append(np.asarray(img))
    return out


def draw_trajectories(frame0, tracks, vis, colors, vis_thresh) -> np.ndarray:
    h, w, _ = frame0.shape
    img = Image.fromarray(frame0).convert("RGB")
    draw = ImageDraw.Draw(img)
    n = tracks.shape[1]
    for pi in range(n):
        col = tuple(int(c) for c in colors[pi])
        pts = [(float(tracks[tj, pi, 0]), float(tracks[tj, pi, 1]))
               for tj in range(tracks.shape[0])
               if vis[tj, pi] >= vis_thresh and 0 <= tracks[tj, pi, 0] < w and 0 <= tracks[tj, pi, 1] < h]
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=1)
    return np.asarray(img)


def main() -> None:
    args = parse_args()
    data = np.load(args.tracks)
    tracks = data["tracks"].astype(np.float32)            # (T, N, 2)
    vis = data["visibility"].astype(np.float32)           # (T, N)
    grid_size = int(data["grid_size"]) if "grid_size" in data else int(round(tracks.shape[1] ** 0.5))

    frames = load_frames(args.video)
    t = min(frames.shape[0], tracks.shape[0])
    frames, tracks, vis = frames[:t], tracks[:t], vis[:t]

    tracks, vis = subsample(tracks, vis, grid_size, args.stride)
    colors = grid_colors(grid_size, args.stride)
    if colors.shape[0] != tracks.shape[1]:  # layout fallback: cycle a rainbow
        colors = grid_colors(int(round(tracks.shape[1] ** 0.5)) or 1, 1)[:tracks.shape[1]]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    overlay = draw_overlay(frames, tracks, vis, colors, args.tail, args.radius, args.vis_thresh)
    mp4_path = args.out.with_name(args.out.name + "_overlay.mp4")
    imageio.mimsave(str(mp4_path), overlay, fps=args.fps, macro_block_size=1)

    traj = draw_trajectories(frames[0], tracks, vis, colors, args.vis_thresh)
    png_path = args.out.with_name(args.out.name + "_trajectories.png")
    imageio.imwrite(str(png_path), traj)

    visible_frac = float((vis >= args.vis_thresh).mean())
    print(f"[viz] frames={t} points_drawn={tracks.shape[1]} (grid {grid_size}x{grid_size}, stride {args.stride}) "
          f"mean_visible={visible_frac:.2f}")
    print(f"[viz] wrote {mp4_path}")
    print(f"[viz] wrote {png_path}")


if __name__ == "__main__":
    main()
