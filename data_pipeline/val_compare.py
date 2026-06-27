# SPDX-License-Identifier: Apache-2.0
"""Re-run validation as a clean GT-vs-generated side-by-side (no track overlay).

For each clip: decode the GT clip, generate with its GT tracks, and write a
left=GT / right=generated side-by-side video + a first-frame comparison PNG, so
the first-frame match and overall reconstruction are easy to eyeball.
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import trackwan_infer as twi  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml")
    p.add_argument("--data", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/"
                   "wan22_a14b_720p_24fps/preprocessed_i2v_track/combined_parquet_dataset")
    p.add_argument("--out", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/val_compare_4k")
    p.add_argument("--fig", default="research_log/figures")
    p.add_argument("--clips", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig, exist_ok=True)

    model, tc = twi.load_trackwan(args.export, args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    ratio = int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
    samples = twi.load_conditioning_from_parquet(args.data, args.clips, text_len)

    for ci, s in zip(args.clips, samples):
        Tpx = (s["first_frame_latent"].shape[2] - 1) * ratio + 1
        gt = twi.decode_reference(model, s["vae_latent"])                       # [T,H,W,3]
        lat = twi.generate(model, first_frame_latent=s["first_frame_latent"],
                           text_embedding=s["text_embedding"], text_attention_mask=s["text_attention_mask"],
                           track_points=s["track_points"][:, :Tpx], track_visibility=s["track_visibility"][:, :Tpx],
                           num_steps=args.steps, seed=args.seed)
        gen = twi.decode_to_pixels(model, lat)
        T = min(len(gt), len(gen))
        side = np.concatenate([gt[:T], gen[:T]], axis=2)                        # [T,H,2W,3] (GT|gen)
        imageio.mimsave(os.path.join(args.out, f"val_clip{ci}_GTleft_GENright.mp4"),
                        list(side), fps=24, macro_block_size=1)
        ff = np.concatenate([gt[0], gen[0]], axis=1)
        imageio.imwrite(os.path.join(args.fig, f"val_firstframe_clip{ci}_GTleft_GENright.png"), ff)
        mse0 = float(((gt[0].astype(np.float32) - gen[0].astype(np.float32)) ** 2).mean())
        print(f"[clip {ci}] first-frame MSE(GT,gen)={mse0:8.1f}  -> {args.out}/val_clip{ci}_GTleft_GENright.mp4", flush=True)


if __name__ == "__main__":
    main()
