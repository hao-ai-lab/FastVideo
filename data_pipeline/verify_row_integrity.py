# SPDX-License-Identifier: Apache-2.0
"""Row-integrity check: does every field of a row belong to the SAME clip?

Yesterday's verify_720p_parquet.py compared 720p rows against 480p rows. That proves
the two datasets agree, but both were produced by the same pipeline, so a systematic
mispairing (row named X carrying clip Y's latent/tracks/caption) would appear in both
and cancel out. This checks each row against the ORIGINAL sources instead:

  caption          <- videos2caption.json  (exact string compare, ALL rows)
  track_points     <- tracks/<name>.npz    (exact float compare after normalize)
  track_visibility <- tracks/<name>.npz    (exact)
  object_ids       <- tracks/<name>.npz    (exact)
  track_weights    <- tracks/<name>.npz    (exact)
  vae_latent       -> VAE decode -> PSNR vs clips/<name>.mp4
  first_frame_lat  -> VAE decode -> PSNR vs frame 0 of clips/<name>.mp4
  clip_feature     <- CLIP re-encode of frame 0 of clips/<name>.mp4
  text_embedding   <- T5 re-encode of the manifest caption

Also renders track points onto the DECODED frames: if the tracks belonged to a
different clip they would not sit on moving content of this one.

Usage:
  python data_pipeline/verify_row_integrity.py --num 12 [--out <png>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch

W = "/mnt/lustre/vlm-s4duan"
OUT = f"{W}/openvid_1m/combined_parquet_dataset_720p"
CLIPS = f"{W}/openvid_1m/clips"
TRACKS = f"{W}/openvid_1m/tracks"
MANIFEST = f"{W}/openvid_1m/videos2caption.json"
MODEL = f"{W}/models/trackwan_1.3b_i2v_d64_nobias_init"


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * float(np.log10(255.0 * 255.0 / mse))


def read_clip(path, n):
    fr = []
    for i, f in enumerate(iio.imiter(path, plugin="pyav")):
        if i >= n:
            break
        fr.append(f)
    return np.stack(fr)


def global_caption_check(manifest):
    """ALL rows: file_name -> caption must equal the manifest's caption for that clip."""
    print("=== GLOBAL: caption/metadata vs manifest, ALL rows ===", flush=True)
    cap_of = {it["path"].rsplit(".", 1)[0]: it["cap"][0] for it in manifest}
    fps_of = {it["path"].rsplit(".", 1)[0]: float(it["fps"]) for it in manifest}
    files = sorted(glob.glob(f"{OUT}/**/*.parquet", recursive=True))
    n = bad_cap = bad_meta = 0
    bad_examples = []
    from concurrent.futures import ThreadPoolExecutor

    def one(f):
        return pq.read_table(f, columns=["file_name", "caption", "fps", "num_frames", "width", "height"])

    with ThreadPoolExecutor(32) as ex:
        for k, t in enumerate(ex.map(one, files)):
            for fn, cap, fps, nf, w, h in zip(t.column("file_name").to_pylist(), t.column("caption").to_pylist(),
                                              t.column("fps").to_pylist(), t.column("num_frames").to_pylist(),
                                              t.column("width").to_pylist(), t.column("height").to_pylist()):
                n += 1
                if cap_of.get(fn) != cap:
                    bad_cap += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(fn)
                if fps != fps_of.get(fn) or nf != 31 or w != 720 or h != 1280:
                    bad_meta += 1
            if (k + 1) % 1000 == 0:
                print(f"    ...{k+1}/{len(files)} files, {n} rows", flush=True)
    print(f"  rows checked            : {n}")
    print(f"  caption mismatches      : {bad_cap} {bad_examples if bad_examples else ''}")
    print(f"  fps/num_frames/w/h bad  : {bad_meta}")
    return bad_cap == 0 and bad_meta == 0, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=f"{W}/openvid_1m/_row_integrity_overlay.png")
    ap.add_argument("--skip-global", action="store_true")
    a = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    ok = True
    if not a.skip_global:
        g_ok, _ = global_caption_check(manifest)
        ok &= g_ok

    # --- sample rows spread across the whole run (by shard completion time, so we
    #     cover all three launch epochs: 16-node wave 1, the 12-node run, the final 16-node run)
    dones = sorted(glob.glob(f"{OUT}/shard_*/.done"), key=os.path.getmtime)
    picks = [dones[int(i * (len(dones) - 1) / max(1, a.num - 1))] for i in range(a.num)]
    rng = random.Random(a.seed)

    print(f"\n=== PER-ROW vs ORIGINAL SOURCES ({a.num} clips spread across the run) ===", flush=True)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()

    cap_of = {it["path"].rsplit(".", 1)[0]: it["cap"][0] for it in manifest}
    panels = []
    for d in picks:
        sd = os.path.dirname(d)
        pqs = sorted(glob.glob(f"{sd}/**/*.parquet", recursive=True))
        if not pqs:
            print(f"  {os.path.basename(sd)}: EMPTY shard (expected for the tail) - skip")
            continue
        t = pq.read_table(rng.choice(pqs))
        i = rng.randrange(t.num_rows)
        r = t.slice(i, 1).to_pylist()[0]
        name = r["file_name"]

        # 1) caption vs manifest
        cap_ok = cap_of.get(name) == r["caption"]

        # 2) tracks vs the clip's own npz (exact)
        z = np.load(f"{TRACKS}/{name}.npz")
        tw, th = float(z["width"]), float(z["height"])
        exp_tp = z["tracks"].astype(np.float32)[:121].copy()
        exp_tp[..., 0] /= tw
        exp_tp[..., 1] /= th
        got_tp = np.frombuffer(r["track_points_bytes"], np.float32).reshape(r["track_points_shape"])
        got_vis = np.frombuffer(r["track_visibility_bytes"], np.float32).reshape(r["track_visibility_shape"])
        tp_ok = np.array_equal(got_tp, exp_tp)
        vis_ok = np.array_equal(got_vis, z["visibility"].astype(np.float32)[:121])
        oid_ok = np.array_equal(np.frombuffer(r["object_ids_bytes"], np.float32),
                                z["object_ids"].astype(np.float32)) if "object_ids" in z else None
        twt_ok = np.array_equal(np.frombuffer(r["track_weights_bytes"], np.float32),
                                z["track_weights"].astype(np.float32)) if "track_weights" in z else None

        # 3) latent -> pixels vs the clip's own mp4
        lat = np.frombuffer(r["vae_latent_bytes"], np.float32).reshape(r["vae_latent_shape"]).copy()
        with torch.no_grad():
            px = vae.decode(torch.from_numpy(lat)[None].to("cuda"), return_dict=False)[0]
        dec = ((px[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        src = read_clip(f"{CLIPS}/{name}.mp4", dec.shape[0])
        p_all = psnr(dec[:len(src)], src[:len(dec)])

        # 4) first-frame conditioning latent -> its frame 0 vs the clip's frame 0
        ff = np.frombuffer(r["first_frame_latent_bytes"], np.float32).reshape(r["first_frame_latent_shape"]).copy()
        with torch.no_grad():
            pxf = vae.decode(torch.from_numpy(ff)[None].to("cuda"), return_dict=False)[0]
        f0 = ((pxf[0, :, 0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        p_f0 = psnr(f0, src[0])

        flags = [f"caption={'OK' if cap_ok else 'MISMATCH'}",
                 f"tracks={'exact' if tp_ok else 'MISMATCH'}",
                 f"vis={'exact' if vis_ok else 'MISMATCH'}",
                 f"oid={'exact' if oid_ok else oid_ok}",
                 f"w={'exact' if twt_ok else twt_ok}",
                 f"latentPSNR={p_all:.1f}dB", f"ff0PSNR={p_f0:.1f}dB"]
        good = cap_ok and tp_ok and vis_ok and (oid_ok is not False) and (twt_ok is not False) and p_all > 30
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {name} [{os.path.basename(sd)}] " + " ".join(flags), flush=True)

        # 5) overlay this row's tracks onto the DECODED frames (visual correspondence)
        idxs = [0, len(dec) // 2, len(dec) - 1]
        H, Wd = dec.shape[1], dec.shape[2]
        row = []
        for fi in idxs:
            img = dec[fi].copy()
            pts = got_tp[fi]
            vis = got_vis[fi] > 0.5
            xs = (pts[:, 0] * Wd).astype(int)
            ys = (pts[:, 1] * H).astype(int)
            keep = vis & (xs >= 1) & (xs < Wd - 1) & (ys >= 1) & (ys < H - 1)
            for x, y in zip(xs[keep][::4], ys[keep][::4]):
                img[y - 1:y + 2, x - 1:x + 2] = (0, 255, 0)
            row.append(img)
        panels.append(np.concatenate(row, axis=1))

    if panels:
        hmin = min(p.shape[0] for p in panels)
        iio.imwrite(a.out, np.concatenate([p[:hmin] for p in panels[:6]], axis=0))
        print(f"\n  track-on-decoded-frame overlay -> {a.out}")

    print(f"\n=== ROW INTEGRITY: {'PASS' if ok else 'FAIL'} ===")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
