# SPDX-License-Identifier: Apache-2.0
"""Data-parallel frame-0 segmentation (SAM2.1-b+ everything-mode) for OpenVid-1M.

Mirrors extract_tracks_mp.py: each Slurm task segments manifest[shard::num_shards]
on one pinned GPU and writes object_ids + track_weights into the EXISTING tracks
.npz (adds keys, keeps tracks/visibility). Idempotent — skips npz that already have
both keys, so it is resumable.

Sharding + GPU pinning from Slurm env:
  shard      = SLURM_PROCID     (0..num_shards-1)
  num_shards = SLURM_NTASKS
  gpu        = SLURM_LOCALID % gpus_per_node

aarch64-safe: PyAV frame-0 decode, torch mask resize (no cv2/decord).

    srun --ntasks-per-node=<gpus*procs_per_gpu> ... \
      python data_pipeline/segment_tracks_mp.py --data-dir <root> --videos-subdir clips
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from segment_tracks import object_ids_for_points, lowrank_track_weights  # noqa: E402


def read_frame0(path: str) -> np.ndarray:
    import av
    c = av.open(str(path))
    try:
        for f in c.decode(video=0):
            return f.to_ndarray(format="rgb24")
    finally:
        c.close()
    raise RuntimeError(f"no frames in {path}")


def extract_masks(res, H: int, W: int) -> np.ndarray:
    if not res or res[0].masks is None:
        return np.zeros((0, H, W), bool)
    import torch
    m = res[0].masks.data
    if m.shape[1:] != (H, W):
        m = torch.nn.functional.interpolate(m[None].float(), size=(H, W), mode="nearest")[0]
    return m.cpu().numpy().astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--videos-subdir", default="clips", help="frame-0 source; MUST match track coords (720p clips)")
    ap.add_argument("--tracks-subdir", default="tracks")
    ap.add_argument("--manifest", default="videos2caption.json")
    ap.add_argument("--model", default="sam2.1_b.pt")
    ap.add_argument("--weights-dir", default="/mnt/lustre/vlm-s4duan/models/seg")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--min-area-frac", type=float, default=0.0)
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_NTASKS", "1")))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import torch
    local = int(os.environ.get("SLURM_LOCALID", str(a.shard)))
    gpu = local % a.gpus_per_node
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"

    from ultralytics import FastSAM, SAM
    wp = Path(a.weights_dir) / a.model
    weight = str(wp) if wp.exists() else a.model
    model = (FastSAM if Path(a.model).name.lower().startswith("fastsam") else SAM)(weight)

    data = Path(a.data_dir)
    items = json.loads((data / a.manifest).read_text())
    if a.limit:
        items = items[:a.limit]
    mine = items[a.shard::a.num_shards]
    print(f"[seg {a.shard}/{a.num_shards}] gpu={gpu} localid={local} items={len(mine)}", flush=True)

    t0 = time.time()
    done = skip = notrk = err = 0
    for it in mine:
        stem = Path(it["path"]).stem
        npz_path = Path(it.get("points_path") or (data / a.tracks_subdir / f"{stem}.npz"))
        if not npz_path.exists():
            notrk += 1
            continue
        try:
            d = dict(np.load(npz_path))
            if "object_ids" in d and "track_weights" in d:
                skip += 1
                continue
            vpath = data / a.videos_subdir / it["path"]
            frame0 = read_frame0(str(vpath))
            H, W = frame0.shape[0], frame0.shape[1]
            res = model(frame0, device=device, retina_masks=True, imgsz=a.imgsz,
                        conf=a.conf, iou=a.iou, verbose=False)
            masks = extract_masks(res, H, W)
            if masks.shape[0] and a.min_area_frac > 0:
                areas = masks.reshape(masks.shape[0], -1).sum(1).astype(np.float64)
                masks = masks[(areas / float(H * W)) >= a.min_area_frac]
            tracks = d["tracks"].astype(np.float32)
            oid = object_ids_for_points(masks, tracks[0], H, W)
            d["object_ids"] = oid.astype(np.int64)
            d["n_objects"] = np.int64(masks.shape[0])
            d["track_weights"] = lowrank_track_weights(tracks)
            tmp = npz_path.with_suffix(".seg.tmp.npz")
            np.savez(tmp, **d)
            tmp.replace(npz_path)
            done += 1
            if done % 200 == 0:
                r = done / (time.time() - t0)
                print(f"[seg {a.shard}] {done} done ({r:.2f}/s), skip={skip} notrk={notrk} err={err}", flush=True)
        except Exception as e:  # noqa: BLE001
            err += 1
            print(f"[seg {a.shard}] ERR {stem}: {repr(e)[:120]}", flush=True)
    dt = time.time() - t0
    print(f"[seg {a.shard}] DONE done={done} skip={skip} notrk={notrk} err={err} in {dt:.1f}s "
          f"-> {done / dt if dt > 0 else 0:.3f} clip/s", flush=True)


if __name__ == "__main__":
    main()
