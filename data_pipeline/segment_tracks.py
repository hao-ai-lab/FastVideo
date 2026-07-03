# SPDX-License-Identifier: Apache-2.0
"""Stage 0d: segment frame-0 (FastSAM) and label each CoTracker grid point by object.

Adds an ``object_ids`` array ([N] int, -1 = background/none) to each tracks ``.npz``,
so the trainer's object-coverage sampling can guarantee >=1 track per object. A grid
point is assigned the SMALLEST mask that contains it (most specific object).

Run on a GPU node (FastSAM is light). Idempotent (skips npz that already have object_ids)::

    srun --jobid=<job> --overlap --ntasks=1 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
      .venv/bin/python data_pipeline/segment_tracks.py --data-dir <dataset root>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_frame0(path: str) -> np.ndarray:
    try:
        from decord import VideoReader, cpu
        return VideoReader(path, ctx=cpu(0))[0].asnumpy()  # HxWx3 uint8
    except Exception:  # noqa: BLE001
        import av
        c = av.open(path)
        for f in c.decode(video=0):
            return f.to_ndarray(format="rgb24")
    raise RuntimeError(f"could not read {path}")


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
    p.add_argument("--min-area-frac",
                   type=float,
                   default=0.0,
                   help="drop masks smaller than this fraction of the frame (fights over-segmentation)")
    p.add_argument("--max-masks", type=int, default=0, help="keep only the N largest masks (0 = keep all)")
    p.add_argument("--limit", type=int, default=None)
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
        if "object_ids" in d and "track_weights" in d:
            n_ok += 1
            continue
        frame0 = read_frame0(str(vpath))
        H, W = frame0.shape[0], frame0.shape[1]
        res = model(frame0,
                    device=args.device,
                    retina_masks=True,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False)
        masks = np.zeros((0, H, W), bool)
        if res and res[0].masks is not None:
            masks = res[0].masks.data.cpu().numpy().astype(bool)  # [M,h,w]
            if masks.shape[1:] != (H, W):  # resize masks to frame res if needed
                import cv2
                masks = np.stack([cv2.resize(m.astype(np.uint8), (W, H),
                                             interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]) \
                    if masks.shape[0] else np.zeros((0, H, W), bool)
        # drop tiny masks / cap count to fight FastSAM over-segmentation
        if masks.shape[0] and (args.min_area_frac > 0 or args.max_masks):
            areas = masks.reshape(masks.shape[0], -1).sum(1).astype(np.float64)
            if args.min_area_frac > 0:
                keep = (areas / float(H * W)) >= args.min_area_frac
                masks, areas = masks[keep], areas[keep]
            if args.max_masks and masks.shape[0] > args.max_masks:
                masks = masks[np.argsort(-areas)[:args.max_masks]]
        tracks = d["tracks"].astype(np.float32)  # [T,N,2] px (orig res)
        oid = object_ids_for_points(masks, tracks[0], H, W)  # frame-0 positions
        d["object_ids"] = oid.astype(np.int64)
        d["n_objects"] = np.int64(masks.shape[0])
        d["track_weights"] = lowrank_track_weights(tracks)  # [N] low-rank informativeness in [0,1]
        tmp = npz_path.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(npz_path)
        n_ok += 1
        cov = int((oid >= 0).sum())
        print(
            f"[seg] [{k}/{len(items)}] {vpath.name}: {masks.shape[0]} objs, "
            f"{cov}/{oid.shape[0]} grid pts labeled, "
            f"w[mean={d['track_weights'].mean():.3f} >0.5={(d['track_weights'] > 0.5).mean():.2f}]",
            flush=True)
    print(f"[seg] done; {n_ok}/{len(items)} npz have object_ids", flush=True)


if __name__ == "__main__":
    main()
