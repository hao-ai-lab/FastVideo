# SPDX-License-Identifier: Apache-2.0
"""Stage 0c: extract dense point tracks from generated videos with CoTracker v3.

For each .mp4 produced by ``generate_videos.py`` we run CoTracker v3 (``cotracker3_offline``)
with a ``grid_size``x``grid_size`` regular query grid (default 50x50 = 2500 points) and save
the per-frame tracks + visibility. We then patch ``points_path`` (absolute) into the
manifest so the future points-aware preprocess task can find them (mirrors how MatrixGame2
references ``action_path``).

Tracks are stored in ORIGINAL video pixel coordinates. The full 2500-point grid + visibility
are kept; the trainer samples 1-200 points per step.

If ``--detect-entries`` is set, FastSAM detects objects entering after frame 0. For each
entry event at frame T_entry a second CoTracker pass runs the full grid from T_entry. Grid
points landing on the new object replace dead background slots (those permanently occluded
from T_entry onwards), keeping N = grid_size^2 throughout.

Run on a GPU node (never the login node), e.g.:

    srun --jobid=<shao_wm jobid> --overlap --ntasks=1 \\
        .venv/bin/python data_pipeline/extract_tracks.py \\
        --data-dir /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_t2v_720p \\
        --grid-size 50 --detect-entries

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
    p.add_argument("--index", type=int, nargs="+", default=None, metavar="IDX",
                   help="Only process videos with these indices (e.g. --index 4 7 12).")
    p.add_argument("--rank", type=int, default=0, help="GPU rank for sharding (0-indexed).")
    p.add_argument("--world-size", type=int, default=1, help="Total number of parallel processes.")
    p.add_argument("--force", action="store_true", help="Re-extract even if .npz already exists.")
    p.add_argument("--verbose", action="store_true", help="Print per-mask debug info for entry detection.")
    # Entry-frame detection
    p.add_argument("--detect-entries", action="store_true",
                   help="Detect objects entering after frame 0 and replace dead background slots.")
    p.add_argument("--sam-model", type=str, default="FastSAM-s.pt")
    p.add_argument("--sam-conf", type=float, default=0.75)
    p.add_argument("--sam-iou", type=float, default=0.9)
    p.add_argument("--sam-imgsz", type=int, default=1024)
    p.add_argument("--entry-sample-every", type=int, default=5,
                   help="Check for new objects every N frames.")
    p.add_argument("--entry-new-area", type=float, default=0.3,
                   help="A mask triggers entry detection if at least this fraction of its area "
                        "is not covered by any frame-0 mask (catches partially-entering objects).")
    p.add_argument("--entry-min-area", type=float, default=0.005,
                   help="Min area fraction for a new-object mask to be considered.")
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
    """Return (video[1,T,C,H,W] float 0-255, H, W)."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(path), ctx=cpu(0))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # (T, H, W, C) uint8
    except Exception:  # noqa: BLE001
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
    tracks = pred_tracks[0].float().cpu().numpy()
    vis = pred_vis[0].cpu().numpy()

    if downscale != 1.0:
        tracks[..., 0] *= w / float(sw)
        tracks[..., 1] *= h / float(sh)
    return tracks.astype(np.float32), vis


@torch.no_grad()
def track_with_queries(model, video: torch.Tensor, queries_txy: np.ndarray, downscale: float,
                       device: str, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Track explicit query points. queries_txy: [K,3] as (t, x, y) in original pixel coords."""
    track_video = video
    sh, sw = H, W
    q = queries_txy.astype(np.float32).copy()
    if downscale != 1.0:
        sh = max(1, int(round(H * downscale)))
        sw = max(1, int(round(W * downscale)))
        track_video = F.interpolate(video[0], size=(sh, sw), mode="bilinear", align_corners=False).unsqueeze(0)
        q[:, 1] *= sw / float(W)
        q[:, 2] *= sh / float(H)

    q_tensor = torch.from_numpy(q).unsqueeze(0).to(device)  # [1, K, 3]
    pred_tracks, pred_vis = model(track_video.to(device), queries=q_tensor)
    tracks = pred_tracks[0].float().cpu().numpy()
    vis = pred_vis[0].cpu().numpy()

    if downscale != 1.0:
        tracks[..., 0] *= W / float(sw)
        tracks[..., 1] *= H / float(sh)
    return tracks.astype(np.float32), vis


def make_grid_queries(grid_size: int, H: int, W: int, frame_t: int) -> np.ndarray:
    """Generate a grid_size x grid_size uniform query grid at frame_t. Returns [N,3] (t,x,y)."""
    ys = np.linspace(0, H - 1, grid_size)
    xs = np.linspace(0, W - 1, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    pts_xy = np.column_stack([xx.ravel(), yy.ravel()])  # [N, 2]
    return np.column_stack([np.full(len(pts_xy), float(frame_t)), pts_xy]).astype(np.float32)



def fastsam_masks(sam_model, frame_rgb: np.ndarray, conf: float, iou: float, imgsz: int,
                  H: int, W: int) -> np.ndarray:
    """Run FastSAM on a uint8 HxWx3 RGB frame, return bool masks [M,H,W]."""
    res = sam_model(frame_rgb, device="cuda", retina_masks=True,
                    imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    if not res or res[0].masks is None:
        return np.zeros((0, H, W), bool)
    masks = res[0].masks.data.cpu().numpy().astype(bool)
    if masks.shape[0] and masks.shape[1:] != (H, W):
        import cv2
        masks = np.stack([
            cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            for m in masks
        ])
    return masks


def max_iou_with_set(mask: np.ndarray, others: np.ndarray) -> float:
    """Max IOU of a single [H,W] bool mask against a set [M,H,W]."""
    if others.shape[0] == 0:
        return 0.0
    inter = (mask & others).reshape(others.shape[0], -1).sum(1)
    union = (mask | others).reshape(others.shape[0], -1).sum(1)
    return float(np.where(union > 0, inter / union, 0.0).max())


def detect_entry_events(video: torch.Tensor, sam_model, H: int, W: int, conf: float, iou: float,
                        imgsz: int, sample_every: int, new_area_thresh: float,
                        min_area_frac: float) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Return ({frame_t: new_masks [M,H,W]}, masks0_union [H,W]) for frames where new
    regions appear that weren't covered by any frame-0 mask.

    Uses new-area fraction rather than IOU so partially-entering objects (partially visible
    at frame 0, more visible later) are correctly detected: only the newly visible region
    triggers detection and receives replacement tracks.
    """
    T = video.shape[1]
    min_area_px = int(min_area_frac * H * W)

    frame0 = video[0, 0].permute(1, 2, 0).numpy().astype(np.uint8)
    masks0 = fastsam_masks(sam_model, frame0, conf, iou, imgsz, H, W)
    # Exclude large background masks (table surface, floor, walls) from masks0_union.
    # Only object-sized masks define "known territory" — background covers the whole frame
    # and would suppress detection of the ball moving to a new position on that surface.
    if masks0.shape[0] > 0:
        areas = masks0.reshape(masks0.shape[0], -1).sum(1)
        object_masks0 = masks0[areas < 0.4 * H * W]
        masks0_union = object_masks0.any(axis=0) if object_masks0.shape[0] > 0 else np.zeros((H, W), bool)
    else:
        masks0_union = np.zeros((H, W), bool)

    # Tracks newly-covered regions across entry events to avoid re-detecting the same
    # entering object at multiple sample frames.
    claimed_new = np.zeros((H, W), bool)

    entry_events: dict[int, np.ndarray] = {}
    for frame_t in range(sample_every, T, sample_every):
        frame = video[0, frame_t].permute(1, 2, 0).numpy().astype(np.uint8)
        masks_t = fastsam_masks(sam_model, frame, conf, iou, imgsz, H, W)
        if masks_t.shape[0] == 0:
            continue

        new_masks = []
        for m in masks_t:
            area = int(m.sum())
            if area < min_area_px:
                continue
            # New area: part of this mask not present in ANY frame-0 mask.
            new_region = m & ~masks0_union
            new_area_frac = float(new_region.sum()) / max(area, 1)
            if new_area_frac < new_area_thresh:
                continue
            # Skip if this new region was already claimed by a prior entry event.
            if float((new_region & claimed_new).sum()) / max(int(new_region.sum()), 1) > 0.5:
                continue
            new_masks.append(m)
            # Dilate claimed region by ~2x the object radius so a fast-moving
            # object doesn't re-trigger entry detection on subsequent sample frames.
            radius = int(np.sqrt(int(new_region.sum()) / np.pi))
            dil = max(1, radius * 2)
            ys, xs = np.where(new_region)
            y0, y1 = max(0, ys.min() - dil), min(H, ys.max() + dil + 1)
            x0, x1 = max(0, xs.min() - dil), min(W, xs.max() + dil + 1)
            claimed_new[y0:y1, x0:x1] = True

        if new_masks:
            entry_events[frame_t] = np.stack(new_masks)

    return entry_events, masks0_union


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
    if args.index is not None:
        wanted = {f"vid_{i:06d}.mp4" for i in args.index}
        videos = [v for v in videos if v.name in wanted]
    if args.limit is not None:
        videos = videos[:args.limit]
    if args.world_size > 1:
        videos = videos[args.rank::args.world_size]
    if not videos:
        print(f"[track] no videos found in {videos_dir}", flush=True)
        return

    cotracker = load_cotracker(args.model, args.device)
    print(f"[track] loaded {args.model}; {len(videos)} videos, grid={args.grid_size}x{args.grid_size}", flush=True)

    sam_model = None
    if args.detect_entries:
        from ultralytics import FastSAM
        sam_model = FastSAM(args.sam_model)
        print(f"[track] entry detection enabled: sam={args.sam_model}, "
              f"conf={args.sam_conf}, every={args.entry_sample_every}f", flush=True)

    stem_to_points: dict[str, Path] = {}
    for k, vpath in enumerate(videos, 1):
        out_path = out_dir / f"{vpath.stem}.npz"
        stem_to_points[vpath.stem] = out_path
        if out_path.exists() and not args.force:
            continue
        video, h, w = read_video(vpath)
        tracks, vis = track_one(cotracker, video, args.grid_size, args.downscale, args.device)

        n_replaced = 0
        if sam_model is not None:
            entry_events, masks0_union = detect_entry_events(
                video, sam_model, h, w,
                conf=args.sam_conf, iou=args.sam_iou, imgsz=args.sam_imgsz,
                sample_every=args.entry_sample_every,
                new_area_thresh=args.entry_new_area,
                min_area_frac=args.entry_min_area,
            )
            for frame_t, new_masks in sorted(entry_events.items()):
                for mi, new_mask in enumerate(new_masks):
                    new_region = new_mask & ~masks0_union

                    # Covered slots: original frame-0 grid points whose position at
                    # T_entry falls inside the new object's region.
                    orig_xi = np.clip(tracks[frame_t, :, 0].round().astype(int), 0, w - 1)
                    orig_yi = np.clip(tracks[frame_t, :, 1].round().astype(int), 0, h - 1)
                    covered_dst = np.where(new_region[orig_yi, orig_xi])[0]
                    dead_dst = covered_dst if covered_dst.size > 0 else \
                        np.where((~vis[frame_t:].astype(bool)).all(axis=0))[0]
                    if dead_dst.size == 0:
                        continue

                    # Run full grid from T_entry — same density as frame-0 grid.
                    queries_txy = make_grid_queries(args.grid_size, h, w, frame_t)
                    e_tracks, e_vis = track_with_queries(
                        cotracker, video, queries_txy, args.downscale, args.device, h, w)
                    e_vis[:frame_t] = False  # object didn't exist before entry frame

                    pts = e_tracks[frame_t]
                    xi = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
                    yi = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
                    object_src = np.where(new_region[yi, xi])[0]

                    n = min(len(object_src), len(dead_dst))
                    if n > 0:
                        dst = dead_dst[:n]
                        src = object_src[:n]
                        tracks[:, dst] = e_tracks[:, src]
                        vis[:, dst] = e_vis[:, src]
                        # CoTracker can mark points invisible even at their query frame
                        # when the object is entering from the edge. Force vis=True at
                        # T_entry so segment_tracks sees these as first-visible there.
                        vis[frame_t, dst] = True
                        n_replaced += n
                    if args.verbose:
                        mask_area = int(new_mask.sum())
                        new_region_area = int(new_region.sum())
                        vis_at_entry = int(vis[frame_t, dst].sum()) if n > 0 else 0
                        print(f"  [entry] t={frame_t} mask#{mi}: area={mask_area}px "
                              f"new_region={new_region_area}px "
                              f"object_src={len(object_src)} covered_dst={len(covered_dst)} "
                              f"dead_dst={len(dead_dst)} -> replacing {n}, "
                              f"vis[{frame_t}, dst].sum()={vis_at_entry}", flush=True)

            frames_str = ",".join(str(t) for t in sorted(entry_events)) if entry_events else "none"
            print(f"[track] [{k}/{len(videos)}] {vpath.name}: "
                  f"{len(entry_events)} entry event(s) at frames [{frames_str}], "
                  f"{n_replaced} slots replaced", flush=True)

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
