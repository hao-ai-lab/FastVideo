#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Layer-wise learning-rate (LWLR) bootstrap experiment.

Question: on a 14B random-init WanTrack model, does giving the track pathway
(track_encoder.* + patch_embedding.weight[:, 36:]) a 100x higher LR than the
rest actually let it grow into functional gradient magnitudes within a small
number of steps? Or is the encoder stuck regardless?

Runs a short training loop on real OpenVid samples, tracking weight+grad norms
of the track pathway at each step. Compares against the "known good" magnitude
observed in 1.3B_merged (per-sample track_encoder.proj grad ~0.5).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as st_load

# Distributed init BEFORE importing fastvideo (same pattern as gradient_flow_analysis)
def _init_distributed_1gpu() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    from fastvideo.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


_init_distributed_1gpu()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient_flow_analysis import (  # noqa: E402
    build_model_from_diffusers,
    load_samples,
    build_i2v_cond_concat,
    flow_matching_noisy,
    Sample,
)


TRACK_PATH_PARAMS = (
    "track_encoder.proj.weight",
    "track_encoder.temporal_conv.weight",
    "track_encoder.proj.bias",           # may not exist depending on TRACKWAN_TRACK_BIAS
    "track_encoder.temporal_conv.bias",
)


def split_param_groups(model, high_lr: float, base_lr: float) -> list[dict]:
    """Give track pathway (track_encoder.* + patch_embedding[:, 36:]) a high LR.

    patch_embedding.weight is a SINGLE tensor of shape [hidden, 52, 1, 2, 2] — we can't
    give sub-slice a different LR through standard param groups. Workaround: put the
    whole patch_embedding in a "hybrid" group with high LR (the base 36 channels will
    also move faster, but by only 100x for a short warmup that's small — 100 steps of
    high LR is still << stage-1 4800 steps of normal LR).
    """
    high_group, base_group = [], []
    named = dict(model.named_parameters())
    for n, p in named.items():
        if not p.requires_grad:
            continue
        # High-LR pathway: encoder convs (+ bias if present) + the whole patch_embedding.weight
        is_encoder = n.startswith("track_encoder.")
        is_patch_embed = (n == "patch_embedding.weight")
        if is_encoder or is_patch_embed:
            high_group.append(p)
        else:
            base_group.append(p)
    return [
        {"params": high_group, "lr": high_lr, "name": "track_pathway"},
        {"params": base_group, "lr": base_lr, "name": "dit_body"},
    ]


def snapshot_norms(model, sample_grad_i: int) -> dict:
    """Return weight/grad norms for the diagnostic params."""
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Track params of interest
        keep = (
            n.startswith("track_encoder.")
            or n == "patch_embedding.weight"
        )
        if not keep:
            continue
        w_norm = p.detach().float().norm().item()
        g_norm = p.grad.detach().float().norm().item() if p.grad is not None else 0.0
        # For patch_embedding, split into base/track slices
        if n == "patch_embedding.weight":
            w_base = p.detach()[:, :36].float()
            w_track = p.detach()[:, 36:].float()
            out["patch_embedding.weight[:, :36]"] = {
                "w_norm": w_base.norm().item(), "g_norm": 0.0,
            }
            out["patch_embedding.weight[:, 36:]"] = {
                "w_norm": w_track.norm().item(), "g_norm": 0.0,
            }
            if p.grad is not None:
                g_base = p.grad.detach()[:, :36].float()
                g_track = p.grad.detach()[:, 36:].float()
                out["patch_embedding.weight[:, :36]"]["g_norm"] = g_base.norm().item()
                out["patch_embedding.weight[:, 36:]"]["g_norm"] = g_track.norm().item()
        else:
            out[n] = {"w_norm": w_norm, "g_norm": g_norm}
    return out


def one_step(model, s: Sample, device: torch.device, flow_shift: float,
              from_ctx_ns) -> float:
    """Forward+backward on one sample; grads accumulate."""
    vae = s.vae_latent.unsqueeze(0).to(device, dtype=torch.bfloat16)
    ff = s.first_frame_latent.unsqueeze(0).to(device, dtype=torch.bfloat16)
    clip = s.clip_feature.unsqueeze(0).to(device, dtype=torch.bfloat16)
    text = s.text_embedding.unsqueeze(0).to(device, dtype=torch.bfloat16)
    mask = torch.ones(1, s.text_embedding.shape[0], device=device, dtype=torch.bfloat16)
    tp = s.track_points.unsqueeze(0).to(device, dtype=torch.bfloat16)
    tv = s.track_visibility.unsqueeze(0).to(device, dtype=torch.bfloat16)

    num_latent_t = vae.shape[2]
    expected_T = (num_latent_t - 1) * 4 + 1
    tp = tp[:, :expected_T]
    tv = tv[:, :expected_T]

    sigma_u = float(torch.rand(1).item())
    noise = torch.randn_like(vae, dtype=torch.float32)
    clean_f = vae.float()
    noisy_f, _, ts_val = flow_matching_noisy(clean_f, noise, sigma_u,
                                              flow_shift=flow_shift)
    noisy = noisy_f.to(torch.bfloat16)
    cond20 = build_i2v_cond_concat(ff, vae_temporal_compression=4)
    hs = torch.cat([noisy, cond20], dim=1)
    ts = torch.tensor([ts_val], device=device, dtype=torch.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), from_ctx_ns(current_timestep=ts, attn_metadata=None):
        pred = model(
            hidden_states=hs, encoder_hidden_states=text, encoder_attention_mask=mask,
            timestep=ts, encoder_hidden_states_image=clip,
            track_points=tp, track_visibility=tv, return_dict=False,
        )
        target = noise - clean_f
        loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
    return float(loss.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_random_init")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--batch-accum", type=int, default=4, help="gradient accumulation steps per optim step")
    ap.add_argument("--high-lr", type=float, default=1e-3, help="LR for track pathway")
    ap.add_argument("--base-lr", type=float, default=1e-5, help="LR for the rest of DiT")
    ap.add_argument("--flow-shift", type=float, default=6.0)
    ap.add_argument("--parquet-glob", type=str,
                    default="/mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset/shard_*/**/*.parquet")
    ap.add_argument("--out", type=str,
                    default="/mnt/lustre/vlm-s4duan/gradient_analysis_14b/lwlr_trace.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from fastvideo.forward_context import set_forward_context
    device = torch.device("cuda")

    samples = load_samples(args.parquet_glob, args.n_samples, seed=args.seed)
    print(f"loaded {len(samples)} samples")
    assert len(samples) >= args.batch_accum, "need enough samples for accumulation"

    print(f"loading {args.model} ...")
    model = build_model_from_diffusers(args.model, device)

    # Set up param groups
    param_groups = split_param_groups(model, high_lr=args.high_lr, base_lr=args.base_lr)
    n_high = sum(p.numel() for p in param_groups[0]["params"])
    n_base = sum(p.numel() for p in param_groups[1]["params"])
    print(f"track_pathway params: {n_high/1e6:.2f}M  (LR={args.high_lr:g})")
    print(f"dit_body params:      {n_base/1e6:.2f}M  (LR={args.base_lr:g})")
    optim = torch.optim.AdamW(param_groups, weight_decay=0.0)  # no wd so we see raw drift

    trace: list[dict] = []
    # step 0 (before any update)
    print("\n=== step 0 (init) ===")
    norms0 = snapshot_norms(model, -1)
    for k, v in norms0.items():
        print(f"  {k:<45s}  w={v['w_norm']:.4f}  g={v['g_norm']:.4f}")
    trace.append({"step": 0, "loss": None, "norms": norms0})

    step = 0
    accum = 0
    losses_accum = []
    for outer in range(args.n_steps * args.batch_accum):
        s = samples[outer % len(samples)]
        loss = one_step(model, s, device, args.flow_shift, set_forward_context)
        losses_accum.append(loss)
        accum += 1
        if accum < args.batch_accum:
            continue
        step += 1
        # snapshot BEFORE step (grads are accumulated over the batch)
        gs = snapshot_norms(model, -1)
        optim.step()
        optim.zero_grad(set_to_none=True)
        avg_loss = float(np.mean(losses_accum))
        losses_accum = []
        accum = 0
        if step % 5 == 0 or step <= 5:
            print(f"\n=== step {step} (loss={avg_loss:.4f}) ===")
            for k, v in gs.items():
                marker = ""
                if k == "track_encoder.proj.weight":
                    marker = "  <-- 1.3B_merged saw per_sample_norm ~0.50"
                elif k == "patch_embedding.weight[:, 36:]":
                    marker = "  <-- 1.3B_merged saw per_sample_norm ~0.73"
                print(f"  {k:<45s}  w={v['w_norm']:.4f}  g={v['g_norm']:.4f}{marker}")
        trace.append({"step": step, "loss": avg_loss, "norms": gs})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "trace": trace}, f, indent=2)
    print(f"\nwrote trace -> {args.out}")


if __name__ == "__main__":
    main()
