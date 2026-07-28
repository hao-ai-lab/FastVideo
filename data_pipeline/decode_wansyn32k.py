#!/usr/bin/env python3
"""Decode FastVideo Wan2.2-Syn-121x704x1280_32k parquet latents to 480x832 mp4s.

The HF dataset stores Wan 2.2 5B VAE latents (48-channel, 16x spatial, 4x
temporal). We can't feed these to our Wan 2.1 WanTrack model directly, so we
decode → pixels → downsize to our training resolution 480x832 → save mp4.
Downstream (SAM + CoTracker + i2v_track preprocess) then treats these as if they
were raw synth mp4s.

Idempotent + resumable: skips samples whose mp4 already exists.

Slurm sharded via ``--num-shards``/``--shard``. Each worker processes
``parquets[shard::num_shards]``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from diffusers import AutoencoderKLWan
from PIL import Image


def _resize_frames(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Center-crop-then-resize a (T,H,W,3) uint8 stack to (T,out_h,out_w,3)."""
    T, H, W, _ = frames.shape
    src_ar = W / H
    dst_ar = out_w / out_h
    if src_ar > dst_ar:
        new_w = int(round(H * dst_ar))
        x0 = (W - new_w) // 2
        cropped = frames[:, :, x0:x0 + new_w, :]
    else:
        new_h = int(round(W / dst_ar))
        y0 = (H - new_h) // 2
        cropped = frames[:, y0:y0 + new_h, :, :]
    out = np.empty((T, out_h, out_w, 3), dtype=np.uint8)
    for t in range(T):
        out[t] = np.asarray(Image.fromarray(cropped[t]).resize((out_w, out_h), Image.BILINEAR))
    return out


@torch.no_grad()
def decode_row(vae: AutoencoderKLWan, latent_bytes: bytes, latent_shape: list[int],
               device: torch.device, dtype: torch.dtype) -> np.ndarray:
    """Decode one row's Wan 2.2 5B latent to a [T,704,1280,3] uint8 pixel stack."""
    lat = np.frombuffer(latent_bytes, dtype=np.float32).reshape(latent_shape)
    lat = torch.from_numpy(lat).to(device=device, dtype=dtype).unsqueeze(0)  # [1,C,T,H,W]
    # unnormalize per-channel
    m = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    s = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    lat = lat * s + m
    pix = vae.decode(lat, return_dict=False)[0]  # [1,3,T_out,H*16,W*16] in [-1,1]
    pix = pix.clamp(-1, 1).float()
    pix = ((pix + 1.0) * 127.5).round().to(torch.uint8)
    pix = pix[0].permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,3]
    return pix


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", required=True, help="dir containing train/Part_*/latents_chunk_*.parquet")
    p.add_argument("--vae-dir", required=True, help="Wan 2.2 5B VAE dir")
    p.add_argument("--out-dir", required=True, help="output root; writes videos/vid_*.mp4 + meta/*.json")
    p.add_argument("--out-h", type=int, default=480)
    p.add_argument("--out-w", type=int, default=832)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--limit-rows", type=int, default=None, help="cap rows for smoke test")
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    vids_dir = Path(args.out_dir) / "videos"
    meta_dir = Path(args.out_dir) / "meta"
    vids_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.parquet_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no *.parquet under {args.parquet_dir}")
    my_files = files[args.shard::args.num_shards]
    print(f"[dec {args.shard}/{args.num_shards}] {len(my_files)} parquets", flush=True)

    vae = AutoencoderKLWan.from_pretrained(args.vae_dir, torch_dtype=dtype).to(device).eval()

    n_ok = 0
    n_skip = 0
    n_err = 0
    t0 = time.time()
    row_budget = args.limit_rows or 10**12
    processed_rows = 0
    for fk, f in enumerate(my_files):
        try:
            table = pq.read_table(f)
        except Exception as e:  # skip unreadable parquets
            print(f"[dec {args.shard}] parquet read fail {f}: {e}", flush=True)
            continue
        # Dataset's `id` field only encodes chunk index (not Part), so the same id
        # appears in many Parts. Prefix Part directory so vid_ids are unique.
        part_name = Path(f).parent.name  # e.g. "Part_57"
        rows = table.to_pylist()
        for r in rows:
            if processed_rows >= row_budget:
                break
            processed_rows += 1
            row_key = str(r.get("id") or r.get("file_name") or f"{fk:05d}_{n_ok:04d}")
            vid_id = f"{part_name}_{row_key}"
            vid_path = vids_dir / f"{vid_id}.mp4"
            meta_path = meta_dir / f"{vid_id}.json"
            if vid_path.exists() and meta_path.exists():
                n_skip += 1
                continue
            try:
                pix = decode_row(vae, r["vae_latent_bytes"], list(r["vae_latent_shape"]),
                                 device=device, dtype=dtype)
                resized = _resize_frames(pix, args.out_h, args.out_w)  # [T, out_h, out_w, 3]
                # atomic mp4 write (imageio infers format from extension → keep .mp4)
                tmp = vid_path.parent / f".{vid_path.name}.tmp.mp4"
                imageio.mimsave(str(tmp), resized, fps=args.fps, codec="libx264",
                                quality=7, macro_block_size=1)
                tmp.rename(vid_path)
                meta = {
                    "path": vid_path.name,
                    "id": vid_id,
                    "cap": [str(r.get("caption") or "")],
                    "fps": float(args.fps),
                    "num_frames": int(resized.shape[0]),
                    "resolution": [int(args.out_h), int(args.out_w)],
                    "src_resolution": [int(r.get("height", 704)), int(r.get("width", 1280))],
                    "src_dataset": "FastVideo/Wan2.2-Syn-121x704x1280_32k",
                }
                meta_path.write_text(json.dumps(meta))
                n_ok += 1
                if n_ok % 20 == 0:
                    dt = time.time() - t0
                    rate = n_ok / max(dt, 1e-9)
                    print(f"[dec {args.shard}] ok={n_ok} skip={n_skip} err={n_err} "
                          f"rate={rate:.2f}/s", flush=True)
            except Exception as e:  # skip a bad row, keep going
                n_err += 1
                print(f"[dec {args.shard}] row {vid_id} FAILED: {e}", flush=True)
        if processed_rows >= row_budget:
            break

    print(f"[dec {args.shard}] DONE ok={n_ok} skip={n_skip} err={n_err} "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
