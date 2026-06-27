# SPDX-License-Identifier: Apache-2.0
"""Controllability test for an overfitted TrackWan model.

Disentangles "did it overfit on the video content" from "did it learn to follow
the tracks" by holding the content conditioning (first frame + text) fixed and
varying ONLY the control tracks:

  - gt        : the clip's own tracks            (reconstruction; should match GT)
  - none      : no tracks at all                 (if motion still appears -> content overfit)
  - pan_*/zoom/drag : authored counterfactuals   (if motion follows -> real control)
  - swap      : another clip's tracks            (motion transfer)

For each, it generates a video, overlays the control tracks, saves an mp4, and
computes CoTracker EPE (input tracks vs tracks re-extracted from the generation).
Low EPE on counterfactuals = genuine track control.
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_tracks as st  # noqa: E402
import trackwan_infer as twi  # noqa: E402


def _overlay(frames_thwc: np.ndarray, tracks_norm: np.ndarray, vis: np.ndarray,
             stride: int = 3) -> list[np.ndarray]:
    from fastvideo.train.callbacks.track_validation import (_draw_overlay, _grid_colors, _subsample)
    T, H, W, _ = frames_thwc.shape
    tt = min(T, tracks_norm.shape[0])
    fr, tr, vs = frames_thwc[:tt], tracks_norm[:tt], vis[:tt]
    grid = int(round(tr.shape[1] ** 0.5))
    tr, vs = _subsample(tr, vs, grid, stride)
    colors = _grid_colors(grid, stride)
    if colors.shape[0] != tr.shape[1]:
        colors = _grid_colors(int(round(tr.shape[1] ** 0.5)) or 1, 1)[:tr.shape[1]]
    trpx = tr.copy()
    trpx[..., 0] *= W
    trpx[..., 1] *= H
    return _draw_overlay(fr, trpx, vs, colors, 12, 2, 0.5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True, help="dcp_to_diffusers export dir")
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml")
    p.add_argument("--data", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/"
                   "wan22_a14b_720p_24fps/preprocessed_i2v_track/combined_parquet_dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, tc = twi.load_trackwan(args.export, args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    samples = twi.load_conditioning_from_parquet(args.data, args.clips, text_len)

    from fastvideo.eval.metrics.motion.cotracker_epe.metric import (compute_epe, load_cotracker)
    ct = load_cotracker(model.device)

    # latent T -> pixel T
    num_lat_t = samples[0]["first_frame_latent"].shape[2]
    ratio = int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
    Tpx = (num_lat_t - 1) * ratio + 1

    results = []
    for ci, s in zip(args.clips, samples):
        gt_tracks = s["track_points"][0].numpy()[:Tpx]       # [T,N,2] normalized
        gt_vis = s["track_visibility"][0].numpy()[:Tpx]
        swap = samples[(args.clips.index(ci) + 1) % len(samples)]
        swap_tracks = swap["track_points"][0].numpy()[:Tpx]
        swap_vis = swap["track_visibility"][0].numpy()[:Tpx]

        g = st.make_grid(50)
        pan_t, pan_v = st.pan(g, Tpx, 0.25, 0.0)
        zoom_t, zoom_v = st.zoom(g, Tpx, 1.25)
        drag_t, drag_v = st.drag(g, Tpx, center=(0.5, 0.5), dx=0.3, dy=0.0, radius=0.2)
        drag_v_sparse = st.select_radius(drag_v, g, center=(0.5, 0.5), radius=0.2)

        controls = {
            "gt": (gt_tracks, gt_vis),
            "none": (None, None),
            "pan_right": (pan_t, pan_v),
            "zoom_in": (zoom_t, zoom_v),
            "drag_dense": (drag_t, drag_v),
            "drag_sparse": (drag_t, drag_v_sparse),
            "swap": (swap_tracks, swap_vis),
        }

        # GT reference (decode the real clip)
        ref = twi.decode_reference(model, s["vae_latent"])
        imageio.mimsave(os.path.join(args.out, f"clip{ci}_reference.mp4"),
                        _overlay(ref, gt_tracks, gt_vis), fps=args.fps, macro_block_size=1)

        for name, (tr, vs) in controls.items():
            tp = torch.from_numpy(tr)[None].float() if tr is not None else None
            tv = torch.from_numpy(vs)[None].float() if vs is not None else None
            lat = twi.generate(model, first_frame_latent=s["first_frame_latent"],
                               text_embedding=s["text_embedding"],
                               text_attention_mask=s["text_attention_mask"],
                               track_points=tp, track_visibility=tv,
                               num_steps=args.steps, seed=args.seed)
            frames = twi.decode_to_pixels(model, lat)
            ov_tr = tr if tr is not None else np.zeros((Tpx, 1, 2), np.float32)
            ov_vs = vs if vs is not None else np.zeros((Tpx, 1), np.float32)
            imageio.mimsave(os.path.join(args.out, f"clip{ci}_{name}.mp4"),
                            _overlay(frames, ov_tr, ov_vs), fps=args.fps, macro_block_size=1)

            epe = None
            if tr is not None:
                H, W = frames.shape[1], frames.shape[2]
                tr_px = tr.copy()
                tr_px[..., 0] *= W
                tr_px[..., 1] *= H
                epe = compute_epe(frames, tr_px, vs, ct, model.device)["epe"]
            results.append((ci, name, epe))
            print(f"[clip {ci}] {name:12s} EPE={epe if epe is None else round(epe,2)}", flush=True)

    print("\n=== EPE summary (lower = follows control better) ===")
    for ci, name, epe in results:
        print(f"  clip {ci}  {name:12s}  {('n/a' if epe is None else f'{epe:7.2f} px')}")


if __name__ == "__main__":
    main()
