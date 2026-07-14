# SPDX-License-Identifier: Apache-2.0
"""Data-parallel CoTracker v3 track extraction for large video sets (OpenVid-1M).

CoTracker batching does NOT help (s/video is flat, memory linear), so we scale by
DATA PARALLELISM: launch many single-video workers, several per GPU, across nodes.
Each worker processes video_list[shard::num_shards] on one pinned GPU and is idempotent.

Sharding + GPU pinning come from Slurm env by default:
  shard      = SLURM_PROCID     (0..num_shards-1)
  num_shards = SLURM_NTASKS
  gpu        = SLURM_LOCALID % gpus_per_node
so `srun --ntasks-per-node=(gpus*procs_per_gpu)` gives procs_per_gpu workers per GPU.

Resamples each clip to --fps / --num-frames and resizes to --height x --width (720p)
before tracking, matching the training spec.
"""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HUB = "facebookresearch/co-tracker"


def load_cotracker(device: str):
    local = Path(torch.hub.get_dir()) / (HUB.replace("/", "_") + "_main")
    if local.exists():
        m = torch.hub.load(str(local), "cotracker3_offline", source="local", trust_repo=True)
    else:
        m = torch.hub.load(HUB, "cotracker3_offline", trust_repo=True)
    return m.to(device).eval()


def read_clip(path: str, fps: int, num_frames: int, H: int, W: int,
              min_height: int = 0, aspect: float | None = None, aspect_tol: float = 0.1):
    """Decode (PyAV — decord has no aarch64 wheels, opencv needs GUI libs), resample
    to `fps`, take `num_frames`, resize to HxW. Sequential decode so it is codec-robust
    for OpenVid's varied encodings. Enforces resolution/aspect by probing stream metadata
    (fast, no decode). Returns (1,T,C,H,W) on success, or a reason string
    ('lowres' | 'aspect' | 'short') on reject."""
    import av
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        nh, nw = int(stream.height or 0), int(stream.width or 0)
        if min_height and nh and nh < min_height:
            return "lowres"
        if aspect is not None and nw and nh and abs((nw / nh) - aspect) > aspect_tol:
            return "aspect"
        native = float(stream.average_rate) if stream.average_rate else float(fps)
        step = native / float(fps)
        idxs = [int(round(i * step)) for i in range(num_frames)]
        picked, wi, cur = [], 0, 0
        for frame in container.decode(video=0):
            img = None
            while wi < len(idxs) and idxs[wi] == cur:   # handles repeated idxs (fps up-sample)
                if img is None:
                    img = frame.to_ndarray(format="rgb24")  # (H,W,3) uint8
                picked.append(img); wi += 1
            cur += 1
            if wi >= len(idxs):
                break
    finally:
        container.close()
    if wi < len(idxs):
        return "short"  # video too short at target fps
    arr = np.ascontiguousarray(np.stack(picked))  # (T,H,W,C) RGB uint8
    vid = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).float()  # 1,T,C,h,w
    vid = F.interpolate(vid[0], size=(H, W), mode="bilinear", align_corners=False).unsqueeze(0)
    return vid


@torch.no_grad()
def track(model, video, grid, device):
    tr, vis = model(video.to(device), grid_size=grid)
    return tr[0].float().cpu().numpy(), vis[0].cpu().numpy()


def save_clip(video, path, fps):
    """Write the resampled 121f/720p clip so the training video IS the tracked frames."""
    import imageio
    arr = video[0].permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()  # (T,C,H,W)->(T,H,W,C)
    imageio.mimwrite(path, arr, fps=fps, codec="libx264", quality=7, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-list", required=True, help="Text file: one video path per line.")
    ap.add_argument("--out-dir", required=True, help="Where <stem>.npz tracks are written.")
    ap.add_argument("--grid-size", type=int, default=50)
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_NTASKS", "1")))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--min-height", type=int, default=720, help="skip clips whose native height < this (0=off)")
    ap.add_argument("--aspect", type=float, default=16.0 / 9.0, help="target native W/H aspect")
    ap.add_argument("--aspect-tol", type=float, default=0.12, help="allowed |native_aspect - target| (large=off)")
    ap.add_argument("--clips-dir", default=None, help="also save the resampled 121f/720p clip here (aligned to tracks)")
    a = ap.parse_args()

    local = int(os.environ.get("SLURM_LOCALID", str(a.shard)))
    gpu = local % a.gpus_per_node
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.clips_dir:
        Path(a.clips_dir).mkdir(parents=True, exist_ok=True)

    vids = [l.strip() for l in open(a.video_list) if l.strip()]
    mine = vids[a.shard::a.num_shards]
    model = load_cotracker(device)
    print(f"[shard {a.shard}/{a.num_shards}] gpu={gpu} localid={local} videos={len(mine)}", flush=True)

    t0 = time.time(); done = 0; err = 0; rej = {"short": 0, "lowres": 0, "aspect": 0}
    for vp in mine:
        out = Path(a.out_dir) / f"{Path(vp).stem}.npz"
        if out.exists():
            continue
        try:
            res = read_clip(vp, a.fps, a.num_frames, a.height, a.width,
                            a.min_height, a.aspect, a.aspect_tol)
            if isinstance(res, str):
                rej[res] = rej.get(res, 0) + 1; continue
            tr, vis = track(model, res, a.grid_size, device)
            if a.clips_dir:
                save_clip(res, str(Path(a.clips_dir) / f"{Path(vp).stem}.mp4"), a.fps)
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, tracks=tr.astype(np.float32), visibility=vis, grid_size=a.grid_size,
                     height=a.height, width=a.width, num_frames=tr.shape[0], fps=a.fps)
            tmp.replace(out); done += 1
        except Exception as e:  # noqa: BLE001
            err += 1
            print(f"[shard {a.shard}] ERR {Path(vp).name}: {repr(e)[:110]}", flush=True)
    dt = time.time() - t0
    rate = done / dt if dt > 0 else 0.0
    print(f"[shard {a.shard}] DONE done={done} rej={rej} err={err} in {dt:.1f}s -> {rate:.3f} vid/s", flush=True)


if __name__ == "__main__":
    main()
