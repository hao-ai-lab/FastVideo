# SPDX-License-Identifier: Apache-2.0
"""Diagnose the first-frame I2V conditioning + visualize synthetic tracks.

(1) First-frame conditioning bug: the preprocessing stored ``first_frame_latent``
    using vae.scaling_factor/shift_factor (absent for Wan -> effectively RAW),
    while the model normalizes latents with per-channel latents_mean/std. This
    script decodes the stored conditioning under both interpretations and compares
    to the GT first frame to localize the space mismatch, and shows the current
    model's generated first frame.

(2) Synthetic-track previews: overlays authored controls (pan/zoom/drag/...) on the
    real first frame so you can see what the control signals look like.

Outputs PNGs to --out (default research_log/figures/).
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


def _mse(a, b):
    a = a[:min(len(a), len(b))].astype(np.float32)
    b = b[:len(a)].astype(np.float32)
    return float(((a - b) ** 2).mean())


@torch.no_grad()
def _decode_frame0(model, latent_bcthw, *, already_normalized: bool) -> np.ndarray:
    """latent [1,16,T,H,W] -> frame0 uint8 (H,W,3). If raw, normalize first."""
    from fastvideo.training.training_utils import normalize_dit_input
    x = latent_bcthw.to(model.device, torch.bfloat16)
    if not already_normalized:
        x = normalize_dit_input("wan", x, model.vae)
    px = model.decode_latents(x.permute(0, 2, 1, 3, 4))[0]  # [3,T,H,W] in [0,1]
    f0 = (px[:, 0].clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
    return np.transpose(f0, (1, 2, 0))  # H,W,3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml")
    p.add_argument("--data", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/"
                   "wan22_a14b_720p_24fps/preprocessed_i2v_track/combined_parquet_dataset")
    p.add_argument("--out", default="research_log/figures")
    p.add_argument("--clip", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tc = twi.load_trackwan(args.export, args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    s = twi.load_conditioning_from_parquet(args.data, [args.clip], text_len)[0]

    vae_latent = s["vae_latent"]          # raw GT latents
    ff = s["first_frame_latent"]          # stored conditioning latent

    # GT first frame: vae_latent is raw -> normalize -> decode.
    gt0 = _decode_frame0(model, vae_latent, already_normalized=False)

    # Stored conditioning decoded TWO ways to find its true space:
    #  (a) "model's view": treat stored as already-normalized (decode_latents denorms it).
    #      This is effectively what the model was conditioned on.
    cond_as_norm0 = _decode_frame0(model, ff, already_normalized=True)
    #  (b) "as raw": treat stored as raw -> normalize -> decode.
    cond_as_raw0 = _decode_frame0(model, ff, already_normalized=False)

    # Current model's generated first frame (GT tracks).
    Tpx = (vae_latent.shape[2] - 1) * int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio) + 1
    lat = twi.generate(model, first_frame_latent=ff, text_embedding=s["text_embedding"],
                       text_attention_mask=s["text_attention_mask"],
                       track_points=s["track_points"][:, :Tpx], track_visibility=s["track_visibility"][:, :Tpx],
                       num_steps=args.steps, seed=1000)
    gen0 = twi.decode_to_pixels(model, lat)[0]

    # Save individual + side-by-side.
    panels = {"1_gt_first_frame": gt0,
              "2_cond_as_normalized_MODELVIEW": cond_as_norm0,
              "3_cond_as_raw_then_normalized": cond_as_raw0,
              "4_generated_first_frame": gen0}
    for name, img in panels.items():
        imageio.imwrite(os.path.join(args.out, f"firstframe_clip{args.clip}_{name}.png"), img)
    strip = np.concatenate([gt0, cond_as_norm0, cond_as_raw0, gen0], axis=1)
    imageio.imwrite(os.path.join(args.out, f"firstframe_clip{args.clip}_compare.png"), strip)

    print("=== first-frame diagnosis (clip %d) ===" % args.clip)
    print(f"  MSE(GT, cond-as-normalized / MODEL'S VIEW) = {_mse(gt0, cond_as_norm0):8.1f}")
    print(f"  MSE(GT, cond-as-raw-then-normalized)       = {_mse(gt0, cond_as_raw0):8.1f}")
    print(f"  MSE(GT, generated first frame)             = {_mse(gt0, gen0):8.1f}")
    print("  -> if 'as-raw' MSE << 'model's view' MSE, the stored conditioning is RAW")
    print("     and the model was fed a mis-normalized first frame (the bug).")

    # ---- synthetic track previews on the real first frame ----
    for name in ["pan_right", "zoom_in", "rotate_cw", "drag_center_right", "swirl"]:
        tr, vis = st.preset(name, Tpx, 50, strength=0.25)
        H, W, _ = gt0.shape
        ov = st._overlay_preview(gt0, st.to_pixel(tr, H, W), vis, stride=3)
        imageio.imwrite(os.path.join(args.out, f"trackpreview_{name}.png"), ov)
    print(f"\n[viz] wrote first-frame panels + synthetic track previews to {args.out}/")


if __name__ == "__main__":
    main()
