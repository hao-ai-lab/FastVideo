# SPDX-License-Identifier: Apache-2.0
"""Stage 0d: segment frames (FastSAM) and label each CoTracker grid point by object.

Adds an ``object_ids`` array ([N] int, -1 = background/none) to each tracks ``.npz``,
so the trainer's object-coverage sampling can guarantee >=1 track per object.

Each point is assigned based on its FIRST-VISIBLE frame (from CoTracker visibility):
FastSAM runs on each unique first-visible frame, and the point is assigned the smallest
mask containing its position at that frame. This correctly handles objects that enter
the scene after frame 0.

Run on a GPU node (FastSAM is light). Idempotent (skips npz that already have object_ids)::

    srun --jobid=<job> --overlap --ntasks=1 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
      .venv/bin/python data_pipeline/segment_tracks.py --data-dir <dataset root>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_frame(path: str, idx: int) -> np.ndarray:
    """Read a single frame by index, return HxWx3 uint8."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        return vr[idx].asnumpy()
    except Exception:  # noqa: BLE001
        import av
        c = av.open(path)
        for i, f in enumerate(c.decode(video=0)):
            if i == idx:
                return f.to_ndarray(format="rgb24")
    raise RuntimeError(f"could not read frame {idx} from {path}")


def object_ids_for_points(masks: np.ndarray, pts_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    """masks [M,H,W] bool, pts_xy [N,2] px -> object id per point (-1 none), smallest mask wins."""
    N = pts_xy.shape[0]
    oid = np.full(N, -1, np.int64)
    if masks.size == 0:
        return oid
    areas = masks.reshape(masks.shape[0], -1).sum(1)  # [M]
    order = np.argsort(areas)  # smallest first -> assign, larger won't overwrite
    xi = np.clip(pts_xy[:, 0].round().astype(int), 0, W - 1)
    yi = np.clip(pts_xy[:, 1].round().astype(int), 0, H - 1)
    assigned = np.zeros(N, bool)
    for m in order:
        inside = masks[m][yi, xi] & (~assigned)
        oid[inside] = int(m)
        assigned |= inside
    return oid


def extract_masks(res, H: int, W: int, min_area_frac: float, max_masks: int) -> np.ndarray:
    """Pull masks out of a FastSAM result, resize if needed, and apply filtering."""
    masks = np.zeros((0, H, W), bool)
    if res and res[0].masks is not None:
        masks = res[0].masks.data.cpu().numpy().astype(bool)
        if masks.shape[0] and masks.shape[1:] != (H, W):
            import cv2
            masks = np.stack([
                cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                for m in masks
            ])
    if masks.shape[0] and (min_area_frac > 0 or max_masks):
        areas = masks.reshape(masks.shape[0], -1).sum(1).astype(np.float64)
        if min_area_frac > 0:
            keep = (areas / float(H * W)) >= min_area_frac
            masks, areas = masks[keep], areas[keep]
        if max_masks and masks.shape[0] > max_masks:
            masks = masks[np.argsort(-areas)[:max_masks]]
    return masks


def assign_object_ids_multiframe(
    frame_masks: dict[int, np.ndarray],
    first_visible: np.ndarray,
    tracks: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Assign globally-unique object IDs using each point's first-visible frame.

    frame_masks: {frame_idx: masks [M,H,W] bool}
    first_visible: [N] int, -1 = never visible
    tracks: [T,N,2] px
    Returns object_ids [N] int64, -1 = background/never visible.
    """
    N = first_visible.shape[0]
    oid = np.full(N, -1, np.int64)
    global_offset = 0
    for frame_t, masks in sorted(frame_masks.items()):
        point_sel = first_visible == frame_t
        if point_sel.any() and masks.shape[0] > 0:
            pts_xy = tracks[frame_t, point_sel]
            local_oid = object_ids_for_points(masks, pts_xy, H, W)
            oid[point_sel] = np.where(local_oid >= 0, local_oid + global_offset, -1)
        global_offset += masks.shape[0]
    return oid


def lowrank_track_weights(tracks: np.ndarray, rank: int = 3, pct: float = 97.0) -> np.ndarray:
    """Per-point sampling weight in [0,1] = percentile-normalized low-rank motion residual.

    Stack per-point displacement [N, 2T], subtract the mean trajectory + top-`rank` shared
    SVD modes (camera / dominant scene motion), and take the residual norm. A point is heavy
    iff it moves *uniquely* relative to all other points (independent object motion), not just
    a lot -- so on egocentric/moving-camera clips the head-motion background is down-weighted.
    """
    T, N, _ = tracks.shape
    D = (tracks - tracks[0:1]).transpose(1, 0, 2).reshape(N, 2 * T).astype(np.float64)
    Dc = D - D.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    r = int(min(rank, S.shape[0]))
    resid = np.sqrt(((Dc - (U[:, :r] * S[:r]) @ Vt[:r])**2).sum(1))
    lo, hi = float(resid.min()), float(np.percentile(resid, pct))
    w = np.zeros_like(resid) if hi <= lo else np.clip((resid - lo) / (hi - lo), 0.0, 1.0)
    return w.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--videos-subdir", type=str, default="videos")
    p.add_argument("--tracks-subdir", type=str, default="tracks")
    p.add_argument("--manifest", type=str, default="videos2caption.json")
    p.add_argument("--model", type=str, default="FastSAM-s.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.9)
    p.add_argument("--min-area-frac", type=float, default=0.0,
                   help="drop masks smaller than this fraction of the frame")
    p.add_argument("--max-masks", type=int, default=0, help="keep only the N largest masks (0 = keep all)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="re-run even if object_ids already present")
    args = p.parse_args()

    from ultralytics import FastSAM
    model = FastSAM(args.model)

    manifest_path = args.data_dir / args.manifest
    items = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    if args.limit:
        items = items[:args.limit]
    n_ok = 0
    for k, item in enumerate(items, 1):
        vpath = args.data_dir / args.videos_subdir / item["path"]
        npz_path = Path(item.get("points_path") or (args.data_dir / args.tracks_subdir / f"{vpath.stem}.npz"))
        if not npz_path.exists():
            print(f"[seg] [{k}/{len(items)}] {vpath.name}: no npz, skip", flush=True)
            continue
        d = dict(np.load(npz_path))
        if not args.force and "object_ids" in d and "track_weights" in d:
            n_ok += 1
            continue

        tracks = d["tracks"].astype(np.float32)   # [T,N,2] px
        vis = d["visibility"].astype(bool)         # [T,N]
        H, W = int(d["height"]), int(d["width"])

        # Per-point first-visible frame; -1 for points CoTracker never marks visible
        ever_visible = vis.any(axis=0)                                          # [N]
        first_visible = np.where(ever_visible, np.argmax(vis, axis=0), -1)     # [N]
        unique_frames = sorted(set(first_visible[ever_visible].tolist()))

        # Run FastSAM on each unique first-visible frame
        frame_masks: dict[int, np.ndarray] = {}
        for frame_t in unique_frames:
            frame = read_frame(str(vpath), frame_t)
            res = model(frame, device=args.device, retina_masks=True,
                        imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            frame_masks[frame_t] = extract_masks(res, H, W, args.min_area_frac, args.max_masks)

        oid = assign_object_ids_multiframe(frame_masks, first_visible, tracks, H, W)
        n_objects = int(np.unique(oid[oid >= 0]).shape[0]) if (oid >= 0).any() else 0

        d["object_ids"] = oid.astype(np.int64)
        d["n_objects"] = np.int64(n_objects)
        d["track_weights"] = lowrank_track_weights(tracks)
        tmp = npz_path.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(npz_path)
        n_ok += 1
        cov = int((oid >= 0).sum())
        print(
            f"[seg] [{k}/{len(items)}] {vpath.name}: {n_objects} objs across {len(unique_frames)} frames, "
            f"{cov}/{oid.shape[0]} grid pts labeled, "
            f"w[mean={d['track_weights'].mean():.3f} >0.5={(d['track_weights'] > 0.5).mean():.2f}]",
            flush=True)
    print(f"[seg] done; {n_ok}/{len(items)} npz have object_ids", flush=True)


if __name__ == "__main__":
    main()
