# SPDX-License-Identifier: Apache-2.0
"""Quick motion-CFG test: does guidance>1 improve action adherence on an existing ckpt?"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_tracks as st
import trackwan_infer as twi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument(
        "--data",
        default=
        "/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/droid_track_200/preprocessed_i2v_track/combined_parquet_dataset"
    )
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_droid_overfit.yaml")
    p.add_argument("--clips", type=int, nargs="+", default=[0, 1])
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--steps", type=int, default=30)
    a = p.parse_args()
    model, tc = twi.load_trackwan(a.export, a.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    samples = twi.load_conditioning_from_parquet(a.data, a.clips, text_len)
    from fastvideo.eval.metrics.motion.cotracker_epe.metric import compute_epe, load_cotracker
    ct = load_cotracker(model.device)
    nlt = samples[0]["first_frame_latent"].shape[2]
    ratio = int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
    Tpx = (nlt - 1) * ratio + 1
    # grid-snapped dense drag control (moves a radius-0.2 region by +0.3 x)
    g = st.make_grid(50)
    dt, dv = st.drag(g, Tpx, center=(0.5, 0.5), dx=0.3, dy=0.0, radius=0.2)
    disp = np.sqrt(((dt - dt[0:1])**2).sum(-1))
    moving = disp.max(0) > 8 / 832.0  # normalized thresh ~ px
    mv = np.where(moving)[0]
    print(f"{'clip':>4} {'scale':>6} {'EPE_all':>8} {'EPE_moving':>11}")
    for ci, s in zip(a.clips, samples, strict=False):
        tp = torch.from_numpy(dt)[None].float()
        tvv = torch.from_numpy(dv)[None].float()
        for sc in a.scales:
            lat = twi.generate(model,
                               first_frame_latent=s["first_frame_latent"],
                               text_embedding=s["text_embedding"],
                               text_attention_mask=s["text_attention_mask"],
                               track_points=tp,
                               track_visibility=tvv,
                               clip_feature=s["clip_feature"],
                               num_steps=a.steps,
                               seed=1000,
                               guidance_scale=sc)
            fr = twi.decode_to_pixels(model, lat)
            H, W = fr.shape[1], fr.shape[2]
            dpx = dt.copy()
            dpx[..., 0] *= W
            dpx[..., 1] *= H
            epe_all = compute_epe(fr, dpx, dv, ct, model.device)["epe"]
            epe_mv = compute_epe(fr, dpx[:, mv], dv[:, mv], ct, model.device)["epe"]
            print(f"{ci:>4} {sc:>6.1f} {epe_all:>8.2f} {epe_mv:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
