# SPDX-License-Identifier: Apache-2.0
"""Pre-generate controllability-eval artifacts for the viewer app.

For one checkpoint, over clips x counterfactuals, generate the video and save:
  - <clip>_<ctrl>__gen.mp4      raw generation
  - <clip>_<ctrl>__input.mp4    gen + the INPUT control tracks (what we asked for)
  - <clip>_<ctrl>__tracked.mp4  gen + tracks RE-EXTRACTED from the gen (CoTracker3)
  - <clip>_<ctrl>__heat.mp4     gen + points colored by per-point EPE (green=followed,
                                red=ignored) -- the EPE heatmap (when tracks apply)
plus per clip: <clip>__original.mp4 (real clip) and <clip>__original_tracks.mp4.

A manifest.json records EPE / n_points / coverage and the file names so the viewer
(examples/inference/gradio/trackwan/app_viewer.py) can browse without a GPU.

Run one process per checkpoint (parallel across GPUs)::

    srun ... env CUDA_VISIBLE_DEVICES=5 .venv/bin/python data_pipeline/gen_eval_artifacts.py \
      --export <export_3000> --name "step 3000" --out <artifacts_root>/step3000 \
      --data <funinp_now parquet> --clips 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_tracks as st  # noqa: E402
import trackwan_infer as twi  # noqa: E402
from fastvideo.eval.metrics.motion.control_sensitivity.metric import (  # noqa: E402
    compute_control_sensitivity, render_diff_heatmap)

HEAT_SCALE = 40.0  # px error mapped green(0)->red(>=40)


def _draw(frames: Any, trpx: Any, vis: Any, colors: Any) -> Any:
    from fastvideo.train.callbacks.track_validation import _draw_overlay
    return _draw_overlay(frames, trpx, vis, colors, 9, 2, 0.65)


def _solid(m: Any, rgb: Any) -> np.ndarray:
    return np.tile(np.array([rgb], np.uint8), (m, 1))


def _heat_colors(perpt: Any) -> np.ndarray:
    e = np.clip(np.nan_to_num(perpt, nan=HEAT_SCALE) / HEAT_SCALE, 0.0, 1.0)
    r = (255 * e).astype(np.uint8)
    g = (255 * (1 - e)).astype(np.uint8)
    b = np.full_like(r, 40)
    return np.stack([r, g, b], 1)


def _retrack_core(frames: Any,
                  tr_px: Any,
                  vs: Any,
                  ct: Any,
                  device: Any,
                  max_points: int = 600,
                  seed: int = 0,
                  qf: int = 0,
                  vt: float = 0.5) -> dict[str, Any] | None:
    """Mirror compute_epe but also return the extracted tracks + per-point error."""
    from fastvideo.eval.metrics.motion.cotracker_epe.metric import _retrack
    T = min(frames.shape[0], tr_px.shape[0])
    frames, tr_px, vs = frames[:T], tr_px[:T], vs[:T]
    idx = np.nonzero(vs[qf] > vt)[0]
    if idx.size == 0:
        return None
    if max_points and idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    q_xy = tr_px[qf, idx]
    queries = np.concatenate([np.full((idx.size, 1), qf, np.float32), q_xy], axis=1)
    rt, _ = _retrack(ct, frames, queries, device)  # [T,M,2] px
    Tr = min(T, rt.shape[0])
    tgt = tr_px[:Tr][:, idx]
    pred = rt[:Tr]
    mask = vs[:Tr][:, idx] > vt
    d = np.sqrt(((pred - tgt)**2).sum(-1))  # [Tr,M]
    valid = d[mask]
    epe = float(valid.mean()) if valid.size else None
    perpt = np.array([d[:, m][mask[:, m]].mean() if mask[:, m].any() else np.nan for m in range(idx.size)])
    disp = np.sqrt(((tgt - tgt[0:1])**2).sum(-1))  # input travel from frame 0
    moving = disp.max(0) > 8.0
    mm = mask & moving[None, :]
    epe_moving = float(d[mm].mean()) if d[mm].size else None
    return dict(tgt=tgt,
                pred=pred,
                mask=mask,
                epe=epe,
                epe_moving=epe_moving,
                perpt=perpt,
                n=int(idx.size),
                n_moving=int(moving.sum()),
                cov=float(mask.mean()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--name", required=True, help="checkpoint label e.g. 'step 3000'")
    p.add_argument("--out", required=True)
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--clips", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, tc = twi.load_trackwan(args.export, args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    samples = twi.load_conditioning_from_parquet(args.data, args.clips, text_len)
    from fastvideo.eval.metrics.motion.cotracker_epe.metric import load_cotracker
    ct = load_cotracker(model.device)
    num_lat_t = samples[0]["first_frame_latent"].shape[2]
    ratio = int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
    Tpx = (num_lat_t - 1) * ratio + 1
    dev = model.device

    def save(frames: Any, name: str) -> str:
        path = os.path.join(args.out, name)
        imageio.mimsave(path, frames, fps=args.fps, macro_block_size=1)
        return name

    manifest = {"name": args.name, "clips": {}}
    for ci, s in zip(args.clips, samples, strict=False):
        gt_t = s["track_points"][0].numpy()[:Tpx]
        gt_v = s["track_visibility"][0].numpy()[:Tpx]
        swap = samples[(args.clips.index(ci) + 1) % len(samples)]
        g = st.make_grid(50)
        pan_t, pan_v = st.pan(g, Tpx, 0.25, 0.0)
        zoom_t, zoom_v = st.zoom(g, Tpx, 1.25)
        drag_t, drag_v = st.drag(g, Tpx, center=(0.5, 0.5), dx=0.3, dy=0.0, radius=0.2)
        drag_vs = st.select_radius(drag_v, g, center=(0.5, 0.5), radius=0.2)
        controls = {
            "gt": (gt_t, gt_v),
            "none": (None, None),
            "pan_right": (pan_t, pan_v),
            "zoom_in": (zoom_t, zoom_v),
            "drag_dense": (drag_t, drag_v),
            "drag_sparse": (drag_t, drag_vs),
            "swap": (swap["track_points"][0].numpy()[:Tpx], swap["track_visibility"][0].numpy()[:Tpx]),
        }

        ref = twi.decode_reference(model, s["vae_latent"])
        H, W = ref.shape[1], ref.shape[2]
        gtpx = gt_t.copy()
        gtpx[..., 0] *= W
        gtpx[..., 1] *= H
        entry = {
            "caption":
            s["caption"][:240],
            "H":
            int(H),
            "W":
            int(W),
            "original":
            save(ref, f"clip{ci}__original.mp4"),
            "original_tracks":
            save(_draw(ref, gtpx, gt_v, _solid(gt_t.shape[1], [0, 220, 255])), f"clip{ci}__original_tracks.mp4"),
            "controls": {}
        }

        frames_gt = None
        for cname, (tr, vs) in controls.items():
            tp = torch.from_numpy(tr)[None].float() if tr is not None else None
            tv = torch.from_numpy(vs)[None].float() if vs is not None else None
            lat = twi.generate(model,
                               first_frame_latent=s["first_frame_latent"],
                               text_embedding=s["text_embedding"],
                               text_attention_mask=s["text_attention_mask"],
                               track_points=tp,
                               track_visibility=tv,
                               clip_feature=s["clip_feature"],
                               num_steps=args.steps,
                               seed=args.seed)
            frames = twi.decode_to_pixels(model, lat)
            if cname == "gt":
                frames_gt = frames
            trpx = None
            if tr is not None:
                trpx = tr.copy()
                trpx[..., 0] *= W
                trpx[..., 1] *= H
            rec = {
                "gen": save(frames, f"clip{ci}_{cname}__gen.mp4"),
                "epe": None,
                "epe_moving": None,
                "n_points": 0,
                "coverage": 0.0
            }
            if trpx is not None:
                core = _retrack_core(frames, trpx, vs, ct, dev)
                if core is not None:
                    m = core["tgt"].shape[1]
                    rec["input"] = save(_draw(frames, core["tgt"], core["mask"], _solid(m, [0, 220, 255])),
                                        f"clip{ci}_{cname}__input.mp4")
                    rec["tracked"] = save(_draw(frames, core["pred"], core["mask"], _solid(m, [255, 220, 0])),
                                          f"clip{ci}_{cname}__tracked.mp4")
                    rec["heat"] = save(_draw(frames, core["pred"], core["mask"], _heat_colors(core["perpt"])),
                                       f"clip{ci}_{cname}__heat.mp4")
                    rec.update(epe=core["epe"],
                               epe_moving=core["epe_moving"],
                               n_points=core["n"],
                               n_moving=core["n_moving"],
                               coverage=core["cov"])
            # intervention diff vs the GT-track generation (counterfactual control sensitivity)
            if cname != "gt" and frames_gt is not None:
                sens = compute_control_sensitivity(frames_gt, frames, trpx, vs)
                rec["diff"] = save(render_diff_heatmap(frames_gt, frames), f"clip{ci}_{cname}__diff.mp4")
                sr = sens["sensitivity_roi"]
                rec.update(sensitivity_roi=(round(sr, 4) if sr == sr else None),
                           bg_leakage=round(sens["bg_leakage"], 4),
                           localization_iou=round(sens["localization_iou"], 4),
                           mean_diff=round(sens["mean_diff"], 4))
            entry["controls"][cname] = rec
            print(
                f"[{args.name}] clip{ci} {cname:12s} "
                f"EPE={rec['epe'] if rec['epe'] is None else round(float(rec['epe']),2)} "
                f"EPEmv={rec['epe_moving'] if rec['epe_moving'] is None else round(float(rec['epe_moving']),2)} "
                f"sens={rec.get('sensitivity_roi')} iou={rec.get('localization_iou')}",
                flush=True)
        manifest["clips"][str(ci)] = entry

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("ARTIFACTS_DONE", flush=True)


if __name__ == "__main__":
    main()
