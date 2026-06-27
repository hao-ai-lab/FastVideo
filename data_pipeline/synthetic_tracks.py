# SPDX-License-Identifier: Apache-2.0
"""Author synthetic point-track motion controls for WanTrack.

A "track" is just a per-point (x, y) trajectory over time, so we can *author*
counterfactual motion directly (no CoTracker needed -- that is only for
*extracting* tracks from real video). These authored tracks are the test-time
control signal: feed them to the model and check the generated motion follows.

All coordinates are NORMALIZED to [0, 1] (x/width, y/height) -- the same space
the preprocessing stores and the model trains on. Outputs:
    tracks      float32 [T, N, 2]   normalized (x, y) per point per frame
    visibility  float32 [T, N]      1 = active/visible, 0 = inactive/occluded

Modes mirror MotionStream's control surface:
  - ``pan``    : global translation (camera/scene pan)
  - ``zoom``   : scale toward/away from a center (dolly in/out)
  - ``rotate`` : rotate the grid about a center
  - ``drag``   : move only points inside a region (the sparse "drag" handle),
                 with a smooth spatial falloff; everything else stays put
  - ``swirl``  : rotation whose angle decays with radius
  - ``static`` : no motion (tests "does text+first-frame alone reproduce motion?")

``select_*`` helpers turn a dense field into a SPARSE control by zeroing the
visibility of unselected points -- this is how MotionStream lets you drag just a
few points. (Requires a model trained with point-subsampling aug to work well.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Grid + small helpers
# ----------------------------------------------------------------------
def make_grid(grid_size: int = 50) -> np.ndarray:
    """Frame-0 normalized grid positions, returned as [N, 2] (x, y) in [0, 1].

    Row-major (matches CoTracker's grid and our 50x50 preprocessing), so
    ``reshape(grid_size, grid_size, 2)`` is [row(y), col(x)].
    """
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, grid_size, dtype=np.float32),
        np.linspace(0.0, 1.0, grid_size, dtype=np.float32),
        indexing="ij",
    )
    return np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)  # [N, 2]


def _ramp(num_frames: int, ease: bool = True) -> np.ndarray:
    """Time ramp in [0, 1] over ``num_frames`` (optionally smooth ease-in-out)."""
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    if ease:
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
    return t


def _visibility_from_bounds(tracks: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """1 where a point is inside [(-m), (1+m)] in both axes, else 0."""
    lo, hi = -margin, 1.0 + margin
    inb = (tracks[..., 0] >= lo) & (tracks[..., 0] <= hi) & (tracks[..., 1] >= lo) & (tracks[..., 1] <= hi)
    return inb.astype(np.float32)


# ----------------------------------------------------------------------
# Motion fields  ->  (tracks [T,N,2], visibility [T,N])
# ----------------------------------------------------------------------
def static(grid: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.repeat(grid[None], num_frames, axis=0)
    return tracks, np.ones((num_frames, grid.shape[0]), np.float32)


def pan(grid: np.ndarray, num_frames: int, dx: float, dy: float,
        ease: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Translate all points by (dx, dy) (normalized units) over the clip."""
    r = _ramp(num_frames, ease)[:, None, None]
    tracks = grid[None] + r * np.array([dx, dy], np.float32)[None, None]
    return tracks.astype(np.float32), _visibility_from_bounds(tracks)


def zoom(grid: np.ndarray, num_frames: int, scale: float,
         center: tuple[float, float] = (0.5, 0.5), ease: bool = True
         ) -> tuple[np.ndarray, np.ndarray]:
    """Scale toward (scale<1) or away from (scale>1) ``center`` (dolly)."""
    c = np.array(center, np.float32)[None, None]
    r = _ramp(num_frames, ease)[:, None, None]
    s = 1.0 + (scale - 1.0) * r
    tracks = c + (grid[None] - c) * s
    return tracks.astype(np.float32), _visibility_from_bounds(tracks)


def rotate(grid: np.ndarray, num_frames: int, degrees: float,
           center: tuple[float, float] = (0.5, 0.5), ease: bool = True
           ) -> tuple[np.ndarray, np.ndarray]:
    c = np.array(center, np.float32)[None]
    r = _ramp(num_frames, ease)
    out = np.empty((num_frames, grid.shape[0], 2), np.float32)
    rel = grid - c
    for t in range(num_frames):
        a = np.deg2rad(degrees) * r[t]
        ca, sa = np.cos(a), np.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]], np.float32)
        out[t] = rel @ rot.T + c
    return out, _visibility_from_bounds(out)


def drag(grid: np.ndarray, num_frames: int, *, center: tuple[float, float],
         dx: float, dy: float, radius: float = 0.15, falloff: str = "smooth",
         ease: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Move only points within ``radius`` of ``center`` by (dx, dy).

    ``falloff='smooth'`` weights displacement by a cosine window of the distance
    (handle-like drag); ``'hard'`` moves all in-radius points equally. Points
    outside the radius stay put. Visibility stays 1 for all points (dense field
    with a localized motion) -- use ``select_radius`` to make it *sparse*.
    """
    c = np.array(center, np.float32)[None]
    d = np.linalg.norm(grid - c, axis=-1)  # [N]
    if falloff == "smooth":
        w = np.clip(1.0 - d / max(radius, 1e-6), 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * w)  # smooth 0->1
    else:
        w = (d <= radius).astype(np.float32)
    r = _ramp(num_frames, ease)[:, None, None]
    disp = (w[:, None] * np.array([dx, dy], np.float32)[None])[None]  # [1,N,2]
    tracks = grid[None] + r * disp
    return tracks.astype(np.float32), _visibility_from_bounds(tracks)


def swirl(grid: np.ndarray, num_frames: int, degrees: float,
          center: tuple[float, float] = (0.5, 0.5), radius: float = 0.5,
          ease: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Rotation whose angle decays linearly to 0 at ``radius`` from center."""
    c = np.array(center, np.float32)[None]
    rel = grid - c
    dist = np.linalg.norm(rel, axis=-1)
    decay = np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0)
    rmp = _ramp(num_frames, ease)
    out = np.empty((num_frames, grid.shape[0], 2), np.float32)
    for t in range(num_frames):
        a = np.deg2rad(degrees) * rmp[t] * decay
        ca, sa = np.cos(a), np.sin(a)
        x = rel[:, 0] * ca - rel[:, 1] * sa
        y = rel[:, 0] * sa + rel[:, 1] * ca
        out[t] = np.stack([x, y], -1) + c
    return out, _visibility_from_bounds(out)


# ----------------------------------------------------------------------
# Sparsify (sparse "drag a few points" control)
# ----------------------------------------------------------------------
def select_radius(visibility: np.ndarray, grid: np.ndarray, *,
                  center: tuple[float, float], radius: float) -> np.ndarray:
    """Zero visibility outside ``radius`` of ``center`` (keep only a local handle)."""
    c = np.array(center, np.float32)[None]
    keep = (np.linalg.norm(grid - c, axis=-1) <= radius).astype(np.float32)  # [N]
    return visibility * keep[None]


def select_random(visibility: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Keep only ``k`` random points active (rest visibility 0)."""
    rng = np.random.default_rng(seed)
    n = visibility.shape[1]
    keep = np.zeros(n, np.float32)
    keep[rng.choice(n, size=min(k, n), replace=False)] = 1.0
    return visibility * keep[None]


def select_stride(visibility: np.ndarray, grid_size: int, stride: int) -> np.ndarray:
    """Keep a coarse sub-grid (every ``stride``-th point in both dims) active."""
    n = visibility.shape[1]
    mask2d = np.zeros((grid_size, grid_size), np.float32)
    mask2d[::stride, ::stride] = 1.0
    return visibility * mask2d.reshape(1, n)


# ----------------------------------------------------------------------
# Motion transfer: reuse tracks extracted from a real/other video
# ----------------------------------------------------------------------
def from_npz(npz_path: str, num_frames: int | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """Load tracks from an ``extract_tracks.py`` npz, normalized to [0,1]."""
    data = np.load(npz_path)
    tr = data["tracks"].astype(np.float32)            # [T, N, 2] pixels
    vis = data["visibility"].astype(np.float32)       # [T, N]
    w = float(data["width"]) if "width" in data else 1.0
    h = float(data["height"]) if "height" in data else 1.0
    tr = tr.copy()
    tr[..., 0] /= max(w, 1e-6)
    tr[..., 1] /= max(h, 1e-6)
    if num_frames is not None:
        tr, vis = tr[:num_frames], vis[:num_frames]
    return tr, vis


# ----------------------------------------------------------------------
# Convenience: named presets
# ----------------------------------------------------------------------
def preset(name: str, num_frames: int, grid_size: int = 50, *,
           strength: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Author a named motion preset. ``strength`` scales translations/zoom."""
    g = make_grid(grid_size)
    name = name.lower()
    if name == "static":
        return static(g, num_frames)
    if name in ("pan_right", "pan_left", "pan_up", "pan_down"):
        dx = {"pan_right": strength, "pan_left": -strength}.get(name, 0.0)
        dy = {"pan_down": strength, "pan_up": -strength}.get(name, 0.0)
        return pan(g, num_frames, dx, dy)
    if name == "zoom_in":
        return zoom(g, num_frames, 1.0 + strength)
    if name == "zoom_out":
        return zoom(g, num_frames, 1.0 - strength)
    if name in ("rotate_cw", "rotate_ccw"):
        return rotate(g, num_frames, 30.0 * (1 if name == "rotate_cw" else -1))
    if name == "swirl":
        return swirl(g, num_frames, 60.0)
    if name == "drag_center_right":
        return drag(g, num_frames, center=(0.5, 0.5), dx=strength, dy=0.0, radius=0.2)
    raise ValueError(f"unknown preset {name!r}")


PRESETS = ["static", "pan_right", "pan_left", "pan_up", "pan_down",
           "zoom_in", "zoom_out", "rotate_cw", "rotate_ccw", "swirl",
           "drag_center_right"]


def to_pixel(tracks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalized [0,1] tracks -> pixel coords for overlay/EPE."""
    out = tracks.copy()
    out[..., 0] *= width
    out[..., 1] *= height
    return out


# ----------------------------------------------------------------------
# CLI: overlay a preset on a first frame (sanity check, no model/GPU)
# ----------------------------------------------------------------------
def _overlay_preview(frame0: np.ndarray, tracks_px: np.ndarray, vis: np.ndarray,
                     stride: int = 3) -> np.ndarray:
    import colorsys

    from PIL import Image, ImageDraw
    h, w, _ = frame0.shape
    img = Image.fromarray(frame0).convert("RGB")
    draw = ImageDraw.Draw(img)
    n = tracks_px.shape[1]
    gs = int(round(n ** 0.5))
    keep = np.zeros(n, bool)
    keep.reshape(gs, gs)[::stride, ::stride] = True
    for pi in range(n):
        if not keep[pi]:
            continue
        col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb((pi % gs) / gs, 1.0, 1.0))
        pts = [(float(tracks_px[t, pi, 0]), float(tracks_px[t, pi, 1]))
               for t in range(tracks_px.shape[0]) if vis[t, pi] >= 0.5]
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=1)
        if pts:
            x, y = pts[-1]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(description="Preview a synthetic track preset over a first frame.")
    p.add_argument("--frame", type=Path, required=True, help="First-frame image (png/jpg).")
    p.add_argument("--preset", type=str, default="pan_right", choices=PRESETS)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--grid-size", type=int, default=50)
    p.add_argument("--strength", type=float, default=0.25)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import imageio.v2 as imageio
    frame0 = np.asarray(imageio.imread(args.frame))[..., :3]
    h, w, _ = frame0.shape
    tracks, vis = preset(args.preset, args.num_frames, args.grid_size, strength=args.strength)
    overlay = _overlay_preview(frame0, to_pixel(tracks, h, w), vis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(args.out), overlay)
    print(f"[synthetic-tracks] preset={args.preset} frames={args.num_frames} "
          f"points={tracks.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
