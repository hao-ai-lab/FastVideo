# SPDX-License-Identifier: Apache-2.0
"""Verify a preprocessed i2v_track parquet row by decoding its VAE latent back to pixels.

The preprocess stores the RAW VAE.encode() mean (diffusers' AutoencoderKLWan.decode
consumes raw latents directly; the *training* path is what applies latents_mean/std).
So the check is: raw latent -> vae.decode -> compare to the source clip (PSNR + montage).

Usage:
  python data_pipeline/check_720p_latents.py --parquet-glob '<dir>/**/*.parquet' \
      --clips-dir <clips> --model <model_path> --out <png> [--num 1]
"""
from __future__ import annotations

import argparse
import glob

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * float(np.log10(255.0 * 255.0 / mse))


def _read_clip(path: str, num_frames: int) -> np.ndarray:
    """Source clip -> uint8 [T,H,W,3] (first num_frames)."""
    frames = []
    for i, f in enumerate(iio.imiter(path, plugin="pyav")):
        if i >= num_frames:
            break
        frames.append(f)
    return np.stack(frames)


def _resize_to(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """uint8 [T,H,W,3] -> bilinear-resized uint8, matching the preprocess transform."""
    if frames.shape[1] == h and frames.shape[2] == w:
        return frames
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    return t.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-glob", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--model", default="/mnt/lustre/vlm-s4duan/models/trackwan_1.3b_i2v_d64_nobias_init")
    ap.add_argument("--out", default="/mnt/lustre/vlm-s4duan/openvid_1m/_sanity720/decode_check.png")
    ap.add_argument("--num", type=int, default=1, help="rows to check")
    a = ap.parse_args()

    files = sorted(glob.glob(a.parquet_glob, recursive=True))
    assert files, f"no parquet matched {a.parquet_glob}"

    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()

    rows = []
    for f in files:
        for r in pq.read_table(f).to_pylist():
            rows.append(r)
            if len(rows) >= a.num:
                break
        if len(rows) >= a.num:
            break

    panels = []
    for r in rows:
        lat = np.frombuffer(r["vae_latent_bytes"], np.float32).reshape(r["vae_latent_shape"]).copy()
        z = torch.from_numpy(lat)[None].to("cuda", torch.float32)  # [1,16,T,h,w]
        with torch.no_grad():
            px = vae.decode(z, return_dict=False)[0]  # [1,3,T,H,W] in [-1,1]
        dec = ((px[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()  # [T,H,W,3]
        src = _read_clip(f"{a.clips_dir}/{r['file_name']}.mp4", dec.shape[0])
        # record width/height are (H, W) — the base pipeline stores shape[-2], shape[-1]
        tgt = _resize_to(src, int(r["width"]), int(r["height"]))
        n = min(len(dec), len(tgt))
        p_all = _psnr(dec[:n], tgt[:n])
        p_f0 = _psnr(dec[0], tgt[0])
        print(f"{r['file_name']}: latent {tuple(r['vae_latent_shape'])} -> decoded {dec.shape}, "
              f"src(resized) {tgt.shape} | PSNR all-frames {p_all:.2f} dB, frame0 {p_f0:.2f} dB")
        # montage: [src | decoded] for frames 0, T/2, T-1
        idxs = [0, n // 2, n - 1]
        panels.append(np.concatenate([np.concatenate([tgt[i], dec[i]], axis=1) for i in idxs], axis=0))

    iio.imwrite(a.out, np.concatenate(panels, axis=0))
    print(f"wrote montage (left=source, right=VAE round-trip; rows = frame 0 / mid / last) -> {a.out}")


if __name__ == "__main__":
    main()
