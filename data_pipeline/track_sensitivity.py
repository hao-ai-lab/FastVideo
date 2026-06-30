# SPDX-License-Identifier: Apache-2.0
"""Single-checkpoint *swap sensitivity* (one process per checkpoint).

Linearized, generation-free version of the controllability EPE test: at a fixed
noise + timestep, measure how much the predicted velocity ``v`` moves when we swap
ONLY one conditioner and hold the rest fixed:

  S_track = ||v(swap tracks)      - v(base)|| / ||v(base)||
  S_ff    = ||v(swap first-frame) - v(base)|| / ||v(base)||
  S_text  = ||v(zero text)        - v(base)|| / ||v(base)||

Averaged over a few timesteps and the given clips. This captures the *backbone's*
responsiveness to each input (unlike weight-norm or input-projection magnitude,
which only see the input side). Expectation for the control50 run: S_track high at
step 500, decaying by 6000 (control prior forgotten), while S_ff stays high.

One checkpoint per process (loads via the framework loader, so weights/DTensor are
handled correctly). Launch once per --step across GPUs.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(__file__))
import trackwan_infer as twi  # noqa: E402


def _v(model: Any, latents: torch.Tensor, ff: torch.Tensor, tp: torch.Tensor, tv: torch.Tensor, txt: torch.Tensor,
       mask: torch.Tensor, img: torch.Tensor, ts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    from fastvideo.forward_context import set_forward_context
    cond20 = model._build_i2v_cond_concat(ff)
    model_in = torch.cat([latents.to(dtype), cond20], dim=1)
    with torch.no_grad(), torch.autocast(model.device.type, dtype=dtype), \
            set_forward_context(current_timestep=ts, attn_metadata=None):
        v = model.transformer(hidden_states=model_in,
                              encoder_hidden_states=txt,
                              encoder_attention_mask=mask,
                              timestep=ts,
                              encoder_hidden_states_image=img,
                              track_points=tp,
                              track_visibility=tv,
                              return_dict=False)
    return v.float()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True, help="single export dir (export_<step>)")
    p.add_argument("--step", type=int, default=-1, help="label only")
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml")
    p.add_argument("--data",
                   default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/"
                   "wan22_a14b_720p_24fps/preprocessed_i2v_track_funinp/combined_parquet_dataset")
    p.add_argument("--clips", type=int, nargs="+", default=[0, 1])
    p.add_argument("--timestep-idx", type=int, nargs="+", default=[2, 15, 27])
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    dtype = torch.bfloat16
    model, tc = twi.load_trackwan(args.export, args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    samples = twi.load_conditioning_from_parquet(args.data, args.clips, text_len)
    device = model.device

    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler, )
    sched = FlowMatchEulerDiscreteScheduler(shift=float(model.timestep_shift))
    sched.set_timesteps(30, device=device)
    tlist = [sched.timesteps[i].reshape(1).to(device, dtype) for i in args.timestep_idx]

    pairs = []
    for k, ci in enumerate(args.clips):
        s = samples[k]
        sw = samples[(k + 1) % len(samples)]
        _, _, T, H, W = s["first_frame_latent"].shape
        g = torch.Generator(device="cpu").manual_seed(args.seed + ci)
        latents = torch.randn((1, 16, T, H, W), generator=g, dtype=torch.float32).to(device)

        def d(x: torch.Tensor) -> torch.Tensor:
            return x.to(device, dtype)

        pairs.append(
            dict(lat=latents,
                 ff=d(s["first_frame_latent"]),
                 ff_sw=d(sw["first_frame_latent"]),
                 tp=d(s["track_points"]),
                 tv=d(s["track_visibility"]),
                 tp_sw=d(sw["track_points"]),
                 tv_sw=d(sw["track_visibility"]),
                 txt=d(s["text_embedding"]),
                 mask=d(s["text_attention_mask"]),
                 img=d(s["clip_feature"])))

    st = sf = sx = 0.0
    n = 0
    for pr in pairs:
        for ts in tlist:
            vb = _v(model, pr["lat"], pr["ff"], pr["tp"], pr["tv"], pr["txt"], pr["mask"], pr["img"], ts, dtype)
            nb = vb.norm() + 1e-6
            vt = _v(model, pr["lat"], pr["ff"], pr["tp_sw"], pr["tv_sw"], pr["txt"], pr["mask"], pr["img"], ts, dtype)
            vf = _v(model, pr["lat"], pr["ff_sw"], pr["tp"], pr["tv"], pr["txt"], pr["mask"], pr["img"], ts, dtype)
            vx = _v(model, pr["lat"], pr["ff"], pr["tp"], pr["tv"], torch.zeros_like(pr["txt"]), pr["mask"], pr["img"],
                    ts, dtype)
            st += float((vt - vb).norm() / nb)
            sf += float((vf - vb).norm() / nb)
            sx += float((vx - vb).norm() / nb)
            n += 1
    print(f"RESULT step={args.step} S_track={st/n:.4f} S_ff={sf/n:.4f} S_text={sx/n:.4f}", flush=True)


if __name__ == "__main__":
    main()
