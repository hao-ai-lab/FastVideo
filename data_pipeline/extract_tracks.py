# SPDX-License-Identifier: Apache-2.0
"""Stage 0c: extract dense point tracks from generated videos with CoTracker v3.

For each .mp4 produced by ``generate_videos.py`` we run CoTracker v3 (``cotracker3_offline``)
with a ``grid_size``x``grid_size`` regular query grid (default 50x50 = 2500 points) and save
the per-frame tracks + visibility. We then patch ``points_path`` (absolute) into the
manifest so the future points-aware preprocess task can find them (mirrors how MatrixGame2
references ``action_path``).

Tracks are stored in ORIGINAL video pixel coordinates. The full 2500-point grid + visibility
are kept; the trainer samples 1-200 points per step.

Run on a GPU node (never the login node), e.g.:

    srun --jobid=<shao_wm jobid> --overlap --ntasks=1 \
        .venv/bin/python data_pipeline/extract_tracks.py \
        --data-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_t2v_720p \
        --grid-size 50

torch.hub note: prefetch once on the login node (internet) so the shared cache is warm:
    .venv/bin/python -c "import torch; torch.hub.load('facebookresearch/co-tracker','cotracker3_offline')"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HUB_REPO = "facebookresearch/co-tracker"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="Dataset root from generate_videos.py.")
    p.add_argument("--videos-subdir", type=str, default="videos")
    p.add_argument("--out-subdir", type=str, default="tracks")
    p.add_argument("--manifest", type=str, default="videos2caption.json")
    p.add_argument("--grid-size", type=int, default=50, help="NxN query grid (N*N points).")
    p.add_argument("--model", type=str, default="cotracker3_offline")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--downscale", type=float, default=1.0,
                   help="Run tracking at this spatial scale (coords rescaled back to original px). "
                        "Use <1.0 (e.g. 0.5) if full-res OOMs.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N videos (for smoke tests).")
    return p.parse_args()


def load_cotracker(model_name: str, device: str):
    """Load CoTracker via torch.hub, preferring the warm local cache (offline-safe)."""
    hub_dir = Path(torch.hub.get_dir())
    local = hub_dir / (HUB_REPO.replace("/", "_") + "_main")
    try:
        if local.exists():
            model = torch.hub.load(str(local), model_name, source="local", trust_repo=True)
        else:
            model = torch.hub.load(HUB_REPO, model_name, trust_repo=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load CoTracker ({e!r}). On a node without internet, prefetch on the "
            "login node first: .venv/bin/python -c \"import torch; "
            "torch.hub.load('facebookresearch/co-tracker','cotracker3_offline')\""
        ) from e
    return model.to(device).eval()


def read_video(path: Path) -> tuple[torch.Tensor, int, int]:
    """Return (video[1,T,C,H,W] float 0-255, H, W).

    Recent torchvision dropped ``torchvision.io.read_video``; use decord (fast,
    reliable), falling back to imageio's ffmpeg reader.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(path), ctx=cpu(0))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # (T, H, W, C) uint8
    except Exception:  # noqa: BLE001 - fall back to ffmpeg
        import imageio
        reader = imageio.get_reader(str(path), format="ffmpeg")
        frames = np.stack([np.asarray(f) for f in reader], axis=0)
        reader.close()
    vid = torch.from_numpy(np.ascontiguousarray(frames))
    h, w = int(vid.shape[1]), int(vid.shape[2])
    video = vid.permute(0, 3, 1, 2).unsqueeze(0).float()  # (1,T,C,H,W)
    return video, h, w


@torch.no_grad()
def track_one(model, video: torch.Tensor, grid_size: int, downscale: float, device: str
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return tracks (T,N,2) in original pixel coords and visibility (T,N)."""
    _, t, c, h, w = video.shape
    track_video = video
    if downscale != 1.0:
        sh, sw = max(1, int(round(h * downscale))), max(1, int(round(w * downscale)))
        track_video = F.interpolate(video[0], size=(sh, sw), mode="bilinear", align_corners=False).unsqueeze(0)
    else:
        sh, sw = h, w

    pred_tracks, pred_vis = model(track_video.to(device), grid_size=grid_size)
    tracks = pred_tracks[0].float().cpu().numpy()      # (T, N, 2) in tracking-res px
    vis = pred_vis[0].cpu().numpy()                    # (T, N)

    if downscale != 1.0:  # rescale coords back to original resolution
        tracks[..., 0] *= w / float(sw)
        tracks[..., 1] *= h / float(sh)
    return tracks.astype(np.float32), vis


def patch_manifest(manifest_path: Path, stem_to_points: dict[str, Path]) -> int:
    if not manifest_path.exists():
        return 0
    items = json.loads(manifest_path.read_text())
    patched = 0
    for item in items:
        stem = Path(item.get("path", "")).stem
        pts = stem_to_points.get(stem)
        if pts is not None:
            item["points_path"] = str(pts.resolve())
            patched += 1
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=2))
    tmp.replace(manifest_path)
    return patched


def main() -> None:
    args = parse_args()
    videos_dir = args.data_dir / args.videos_subdir
    out_dir = args.data_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob("*.mp4"))
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        print(f"[track] no videos found in {videos_dir}", flush=True)
        return

    model = load_cotracker(args.model, args.device)
    print(f"[track] loaded {args.model}; {len(videos)} videos, grid={args.grid_size}x{args.grid_size}", flush=True)

    stem_to_points: dict[str, Path] = {}
    for k, vpath in enumerate(videos, 1):
        out_path = out_dir / f"{vpath.stem}.npz"
        stem_to_points[vpath.stem] = out_path
        if out_path.exists():
            continue
        video, h, w = read_video(vpath)
        tracks, vis = track_one(model, video, args.grid_size, args.downscale, args.device)
        # np.savez appends ".npz" unless the name already ends in it, so keep the
        # tmp name ".npz"-terminated to make the atomic replace below match.
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez(
            tmp, tracks=tracks, visibility=vis,
            grid_size=args.grid_size, height=h, width=w,
            num_frames=tracks.shape[0],
        )
        tmp.replace(out_path)
        print(f"[track] [{k}/{len(videos)}] {vpath.name} -> {out_path.name} "
              f"tracks={tracks.shape} vis={vis.shape}", flush=True)

    n = patch_manifest(args.data_dir / args.manifest, stem_to_points)
    print(f"[track] done; patched points_path into {n} manifest entries", flush=True)


if __name__ == "__main__":
    main()
