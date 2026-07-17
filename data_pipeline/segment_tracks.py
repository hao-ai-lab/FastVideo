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


def read_all_frames(path: str) -> np.ndarray:
    """Read all frames from a video, return [T,H,W,3] uint8."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        return vr.get_batch(list(range(len(vr)))).asnumpy()
    except Exception:  # noqa: BLE001
        import av
        c = av.open(path)
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def _colors(n: int) -> np.ndarray:
    import colorsys
    return np.array(
        [[int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 1.0)]
         for i in range(max(1, n))],
        np.uint8,
    )


def render_viz(frames: np.ndarray, tracks: np.ndarray, vis: np.ndarray,
               object_ids: np.ndarray, out_path: Path, fps: int = 24) -> None:
    import imageio.v2 as imageio
    from fastvideo.train.callbacks.track_validation import _draw_overlay

    objs = sorted(int(o) for o in np.unique(object_ids) if int(o) >= 0)
    ocols = _colors(len(objs) + 1)
    N = tracks.shape[1]
    pcols = np.tile(np.array([[110, 110, 110]], np.uint8), (N, 1))
    for oi, o in enumerate(objs):
        pcols[object_ids == o] = ocols[oi % len(ocols)]

    G = int(round(N**0.5))
    if G * G == N:
        k = max(1, G // 50)
        sel = np.arange(N).reshape(G, G)[::k, ::k].reshape(-1)
    else:
        st = max(1, N // 1500)
        sel = np.arange(0, N, st)

    ov = _draw_overlay(frames, tracks[:, sel].copy(), vis[:, sel], pcols[sel], 12, 2, 0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), ov, fps=fps, macro_block_size=1)


def render_seg_viz(frame: np.ndarray, masks: np.ndarray, out_path: Path) -> None:
    import imageio.v2 as imageio
    img = frame.copy()
    if masks.shape[0] > 0:
        colors = _colors(masks.shape[0])
        for i, mask in enumerate(masks):
            img[mask] = (img[mask] * 0.5 + colors[i % len(colors)] * 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_path), img)


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


def extract_masks(result, H: int, W: int, min_area_frac: float, max_masks: int) -> np.ndarray:
    """Pull masks out of a single FastSAM Result, resize if needed, and apply filtering."""
    masks = np.zeros((0, H, W), bool)
    if result is not None and result.masks is not None:
        masks = result.masks.data.cpu().numpy().astype(bool)
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


def masks_for_frames(model, frames: np.ndarray, frame_ts: list[int], cache: dict[int, np.ndarray],
                     args) -> dict[int, np.ndarray]:
    """Segment the requested frames in batched FastSAM forwards, filling/reusing `cache`."""
    todo = [t for t in frame_ts if t not in cache]
    for s in range(0, len(todo), max(1, args.sam_batch)):
        chunk = todo[s:s + max(1, args.sam_batch)]
        res = model([frames[t] for t in chunk], device=args.device, retina_masks=True,
                    imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
        for t, r in zip(chunk, res):
            cache[t] = extract_masks(r, frames.shape[1], frames.shape[2],
                                     args.min_area_frac, args.max_masks)
    return cache


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


def segment_tracks_arrays(tracks: np.ndarray, vis: np.ndarray, H: int, W: int, get_masks,
                          vis_override_every: int, verbose: bool = False
                          ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, int, list[int]]:
    """Core of Stage 4: object IDs from first-visible frames, vis override, low-rank weights.

    Shared by segment_tracks.py and extract_tracks.py --segment (fused mode) so both paths
    produce identical results. ``get_masks(frame_ts)`` must return {frame_t: masks [M,H,W] bool}.
    tracks: [T,N,2] px. vis: [T,N] bool, updated in place by the override sweep.
    Returns (object_ids, n_objects, vis, track_weights, n_overrides, unique_frames).
    """
    T = tracks.shape[0]
    # Per-point first-visible frame; -1 for points CoTracker never marks visible
    ever_visible = vis.any(axis=0)                                          # [N]
    first_visible = np.where(ever_visible, np.argmax(vis, axis=0), -1)     # [N]
    unique_frames = sorted(set(first_visible[ever_visible].tolist()))

    if verbose:
        fv_counts = {f: int((first_visible == f).sum()) for f in unique_frames}
        print(f"  [seg] first_visible frames: {fv_counts}", flush=True)

    # FastSAM masks for each unique first-visible frame
    frame_masks = get_masks(unique_frames)
    if verbose:
        for frame_t, masks in frame_masks.items():
            areas = masks.reshape(masks.shape[0], -1).sum(1).tolist() if masks.shape[0] else []
            print(f"  [seg] frame {frame_t}: {masks.shape[0]} masks, areas={[int(a) for a in areas]}", flush=True)

    oid = assign_object_ids_multiframe(frame_masks, first_visible, tracks, H, W)
    n_objects = int(np.unique(oid[oid >= 0]).shape[0]) if (oid >= 0).any() else 0

    if verbose:
        for frame_t in unique_frames:
            pt_sel = first_visible == frame_t
            labeled = int((oid[pt_sel] >= 0).sum())
            print(f"  [seg] frame {frame_t}: {pt_sel.sum()} pts, {labeled} got oid>=0", flush=True)

    # Vis override: run FastSAM every N frames and set vis=True for object points
    # that fall inside any mask — fixes CoTracker vis=0 on edge-of-frame objects.
    n_overrides = 0
    if vis_override_every > 0 and (oid >= 0).any():
        object_pts = np.where(oid >= 0)[0]
        override_ts = list(range(0, T, vis_override_every))
        override_masks = get_masks(override_ts)
        for frame_t in override_ts:
            masks = override_masks[frame_t]
            if masks.shape[0] == 0:
                continue
            pts = tracks[frame_t, object_pts]                              # [K,2]
            xi = np.clip(pts[:, 0].round().astype(int), 0, W - 1)
            yi = np.clip(pts[:, 1].round().astype(int), 0, H - 1)
            in_any_mask = masks[:, yi, xi].any(axis=0)                    # [K] bool
            newly_visible = in_any_mask & ~vis[frame_t, object_pts]
            vis[frame_t, object_pts] |= in_any_mask
            n_overrides += int(newly_visible.sum())

        # Fill gaps between True frames caused by the sampling interval.
        # If vis is True at frame T and True again at T+k (k <= override_every),
        # the frames in between should also be True — the object didn't disappear.
        V = vis[:, object_pts]                                             # [T,K]
        t_idx = np.arange(T)[:, None]
        prev = np.maximum.accumulate(np.where(V, t_idx, -1), axis=0)
        nxt = np.minimum.accumulate(np.where(V, t_idx, 2 * T)[::-1], axis=0)[::-1]
        fill = (~V) & (prev >= 0) & (nxt < T) & ((nxt - prev) <= vis_override_every)
        vis[:, object_pts] = V | fill

    weights = lowrank_track_weights(tracks)
    return oid.astype(np.int64), n_objects, vis, weights, n_overrides, unique_frames


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
    p.add_argument("--sam-batch", type=int, default=16,
                   help="Frames per batched FastSAM forward.")
    p.add_argument("--min-area-frac", type=float, default=0.0,
                   help="drop masks smaller than this fraction of the frame")
    p.add_argument("--max-masks", type=int, default=0, help="keep only the N largest masks (0 = keep all)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--index", type=int, nargs="+", default=None, metavar="IDX",
                   help="Only process videos at these manifest indices (e.g. --index 4 7 12).")
    p.add_argument("--rank", type=int, default=0, help="GPU rank for sharding (0-indexed).")
    p.add_argument("--world-size", type=int, default=1, help="Total number of parallel processes.")
    p.add_argument("--force", action="store_true", help="re-run even if object_ids already present")
    p.add_argument("--vis-override-every", type=int, default=0,
                   help="Run FastSAM every N frames and set vis=True for object points inside masks. "
                        "Fixes CoTracker vis=0 on edge-of-frame objects. 0 = disabled.")
    p.add_argument("--viz", action="store_true", help="Render a track-overlay mp4 after each video.")
    p.add_argument("--viz-dir", type=str, default=None,
                   help="Output directory for viz mp4s (default: <data-dir>/viz).")
    p.add_argument("--verbose", action="store_true", help="Print per-frame debug info.")
    args = p.parse_args()
    if args.viz and args.viz_dir is None:
        args.viz_dir = str(args.data_dir / "viz")

    from ultralytics import FastSAM
    model = FastSAM(args.model)

    manifest_path = args.data_dir / args.manifest
    items = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    if args.index is not None:
        items = [items[i] for i in args.index if i < len(items)]
    if args.limit:
        items = items[:args.limit]
    if args.world_size > 1:
        items = items[args.rank::args.world_size]
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

        # Decode the video once; frames are reused for segmentation, vis override, and viz.
        frames = read_all_frames(str(vpath))
        mask_cache: dict[int, np.ndarray] = {}

        def get_masks(frame_ts: list[int], _frames=frames, _cache=mask_cache) -> dict[int, np.ndarray]:
            masks_for_frames(model, _frames, frame_ts, _cache, args)
            return {t: _cache[t] for t in frame_ts}

        oid, n_objects, vis, weights, n_overrides, unique_frames = segment_tracks_arrays(
            tracks, vis, H, W, get_masks, args.vis_override_every, verbose=args.verbose)

        if args.vis_override_every > 0:
            d["visibility"] = vis
        d["object_ids"] = oid
        d["n_objects"] = np.int64(n_objects)
        d["track_weights"] = weights
        tmp = npz_path.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(npz_path)
        n_ok += 1
        if args.viz:
            stem_dir = Path(args.viz_dir) / vpath.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            render_viz(frames[:tracks.shape[0]], tracks, vis, oid, stem_dir / "tracks.mp4",
                       fps=int(item.get("fps", 24)))
            # for label, fidx in [("000", 0), ("mid", T_v // 2), ("last", T_v - 1)]:
            #     frame = frames[fidx]
            #     res = model(frame, device=args.device, retina_masks=True,
            #                 imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            #     fmasks = extract_masks(res, H, W, args.min_area_frac, args.max_masks)
            #     render_seg_viz(frame, fmasks, stem_dir / f"seg_frame{label}.jpg")
            print(f"  viz -> {stem_dir}/", flush=True)

        cov = int((oid >= 0).sum())
        override_str = f", {n_overrides} vis overrides" if args.vis_override_every > 0 else ""
        print(
            f"[seg] [{k}/{len(items)}] {vpath.name}: {n_objects} objs across {len(unique_frames)} frames, "
            f"{cov}/{oid.shape[0]} grid pts labeled{override_str}, "
            f"w[mean={d['track_weights'].mean():.3f} >0.5={(d['track_weights'] > 0.5).mean():.2f}]",
            flush=True)
    print(f"[seg] done; {n_ok}/{len(items)} npz have object_ids", flush=True)


if __name__ == "__main__":
    main()
