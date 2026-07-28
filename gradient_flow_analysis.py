#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Per-sample gradient / SNR analysis of the WanTrack DiT, comparing 4 init/checkpoint variants
(zero_init, random_init, merged, overfit_full) on real OpenVid parquet samples.

For each (model, layer) we record:
  avg_per_sample_norm = mean_i ||g_i||         (typical bs=1 grad magnitude)
  mean_grad_norm      = ||mean_i g_i||          (grad magnitude after bs=N averaging)
  snr                 = mean_grad_norm / avg_per_sample_norm

High SNR => signal survives averaging (universal component).
Low SNR  => signal cancels (sample-specific / noise).

Layers tracked (custom-model names, not safetensors names):
  patch_embedding.proj.weight [:, :36] and [:, 36:]
  track_encoder.proj.weight
  track_encoder.temporal_conv.weight
  blocks.0.to_q.weight              (early self-attn Q)
  blocks.14.ffn.fc_in.weight        (mid FFN in-proj)
  blocks.29.to_out.weight           (late self-attn out)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as st_load

# Init distributed BEFORE any fastvideo import that touches SP world.
def _init_distributed_1gpu() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    from fastvideo.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


_init_distributed_1gpu()

# Now safe to import model classes
from fastvideo.configs.models.dits.trackwan import (  # noqa: E402
    TrackWanVideoArchConfig,
    TrackWanVideoConfig,
)
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.models.dits.trackwan.model import TrackWanTransformer3DModel  # noqa: E402
from fastvideo.models.loader.utils import (  # noqa: E402
    get_param_names_mapping,
    hf_to_custom_state_dict,
)


# ------------------------------------------------------------------ #
# Model set-up
# ------------------------------------------------------------------ #
MODELS = {
    "zero_init":     "/mnt/lustre/vlm-s4duan/models/trackwan_1.3b_i2v_d64_nobias_init",
    "random_init":   "/mnt/lustre/vlm-s4duan/models/trackwan_1.3b_i2v_d64_nobias_random_init",
    "merged":        "/mnt/lustre/vlm-s4duan/models/trackwan_1.3b_i2v_d64_merged_from_overfit",
    "overfit_full":  "/mnt/lustre/vlm-s4duan/models/overfit_2000",
    "fun_control":   "/mnt/lustre/vlm-s4duan/models/trackwan_1.3b_i2v_d64_from_fun_control_init",
    # 14B variants
    "14b_zero_init":      "/mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_nobias_init",
    "14b_random_init":    "/mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_random_init",
    "14b_partial_merged": "/mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_partial_merged",
    # zero-gate init WITH bias (the base the step-A overfit started from — gate exactly 0).
    "14b_zero_init_bias": "/mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_zero_init_bias",
    # fresh 14B base with the FIXED-overfit (step A, ckpt-2000) encoder + gate merged in — this
    # is the candidate stage-1 init; its per-sample track grad decides whether it will train.
    "14b_merged_fixed2000": "/mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_merged_from_fixed2000_bias",
}

# Custom-model parameter names we snapshot per sample.
# Special slicing for patch_embedding is handled below.
TRACK_LAYERS = [
    "patch_embedding.proj.weight",  # will split into base=[:, :36] / track=[:, 36:]
    "track_encoder.proj.weight",
    "track_encoder.temporal_conv.weight",
    "blocks.0.to_q.weight",
    "blocks.14.ffn.fc_in.weight",
    "blocks.29.to_out.weight",
]


def build_model_from_diffusers(model_dir: str, device: torch.device) -> TrackWanTransformer3DModel:
    cfg_path = Path(model_dir) / "transformer" / "config.json"
    with open(cfg_path, "r") as f:
        hf_config = json.load(f)

    arch = TrackWanVideoArchConfig(
        num_attention_heads=int(hf_config["num_attention_heads"]),
        attention_head_dim=int(hf_config["attention_head_dim"]),
        in_channels=int(hf_config["in_channels"]),
        out_channels=int(hf_config["out_channels"]),
        text_dim=int(hf_config.get("text_dim", 4096)),
        freq_dim=int(hf_config.get("freq_dim", 256)),
        ffn_dim=int(hf_config["ffn_dim"]),
        num_layers=int(hf_config["num_layers"]),
        cross_attn_norm=bool(hf_config.get("cross_attn_norm", True)),
        qk_norm=str(hf_config.get("qk_norm", "rms_norm_across_heads")),
        eps=float(hf_config.get("eps", 1e-6)),
        image_dim=int(hf_config.get("image_dim") or 0) or None,
        added_kv_proj_dim=int(hf_config.get("added_kv_proj_dim") or 0) or None,
        rope_max_seq_len=int(hf_config.get("rope_max_seq_len", 1024)),
        track_config=dict(hf_config.get("track_config", {})),
    )
    # The trackwan config has an "in_channels: 52" default, but we still pass hf_config so
    # TrackWanTransformer3DModel picks up the correct track_channels from track_config.
    cfg = TrackWanVideoConfig(arch_config=arch)

    # Ensure downstream consumers see the right in_channels regardless of default
    cfg.arch_config.in_channels = int(hf_config["in_channels"])
    cfg.arch_config.out_channels = int(hf_config["out_channels"])

    model = TrackWanTransformer3DModel(config=cfg, hf_config=hf_config)
    model = model.to(device=device, dtype=torch.bfloat16)

    # --- load weights ---
    st_files = sorted(glob.glob(str(Path(model_dir) / "transformer" / "*.safetensors")))
    assert len(st_files) >= 1, f"no safetensors under {model_dir}/transformer"
    hf_sd: dict[str, torch.Tensor] = {}
    for p in st_files:
        hf_sd.update(st_load(p))

    mapping_fn = get_param_names_mapping(model.param_names_mapping)
    custom_sd, _ = hf_to_custom_state_dict(hf_sd, mapping_fn)

    # Load into model
    model_sd = model.state_dict()
    missing, unexpected = [], []
    for k, v in custom_sd.items():
        if k in model_sd:
            model_sd[k].data.copy_(v.to(model_sd[k].dtype))
        else:
            unexpected.append(k)
    for k in model_sd:
        if k not in custom_sd:
            missing.append(k)
    print(f"[{model_dir}] missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("  unexpected head:", unexpected[:5])
    if missing:
        print("  missing head:", missing[:5])

    # Gradient checkpointing needs the forward context to be re-established during
    # backward recompute; we keep both fwd+bwd wrapped in set_forward_context in
    # run_one_model, so grad_checkpointing is safe.
    model.gradient_checkpointing = True
    model._gradient_checkpointing_func = lambda fn, *args, **kwargs: torch.utils.checkpoint.checkpoint(
        fn, *args, use_reentrant=False, **kwargs
    )

    # FastVideo's linear layers create Parameters with requires_grad=False (FSDP path).
    # For gradient analysis we force all params trainable.
    for p in model.parameters():
        p.requires_grad_(True)

    model.train()  # ensure gradients flow
    return model


# ------------------------------------------------------------------ #
# Parquet loader
# ------------------------------------------------------------------ #
@dataclass
class Sample:
    vae_latent: torch.Tensor          # [16, T, H, W]
    first_frame_latent: torch.Tensor  # [16, T, H, W]
    clip_feature: torch.Tensor        # [257, 1280]
    text_embedding: torch.Tensor      # [L, 4096]
    track_points: torch.Tensor        # [T_full, N, 2]
    track_visibility: torch.Tensor    # [T_full, N]
    object_ids: torch.Tensor          # [N]
    track_weights: torch.Tensor       # [N]


def _decode(row, name) -> torch.Tensor:
    shape = row[f"{name}_shape"]
    data = row[f"{name}_bytes"]
    arr = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(arr)


def load_samples(pattern: str, n: int, seed: int = 0) -> list[Sample]:
    files = sorted(glob.glob(pattern, recursive=True))
    random.Random(seed).shuffle(files)
    samples: list[Sample] = []
    for f in files:
        if len(samples) >= n:
            break
        try:
            tbl = pq.read_table(f)
        except Exception as e:
            print("skip", f, e)
            continue
        for row in tbl.to_pylist():
            try:
                s = Sample(
                    vae_latent=_decode(row, "vae_latent"),
                    first_frame_latent=_decode(row, "first_frame_latent"),
                    clip_feature=_decode(row, "clip_feature"),
                    text_embedding=_decode(row, "text_embedding"),
                    track_points=_decode(row, "track_points"),
                    track_visibility=_decode(row, "track_visibility"),
                    object_ids=_decode(row, "object_ids"),
                    track_weights=_decode(row, "track_weights"),
                )
            except Exception as e:
                print("skip row", e)
                continue
            samples.append(s)
            if len(samples) >= n:
                break
    print(f"loaded {len(samples)} samples from {pattern}")
    return samples


# ------------------------------------------------------------------ #
# I2V cond concat and flow-matching noise
# ------------------------------------------------------------------ #
def build_i2v_cond_concat(first_frame_latent: torch.Tensor,
                           vae_temporal_compression: int = 4) -> torch.Tensor:
    """[B,16,T,H,W] first-frame latent -> [B,20,T,H,W] (4 mask + 16 latent).

    Mirrors WanTrackModel._build_i2v_cond_concat (mask=1 on first frame slot only).
    """
    b, _, num_latent_t, h, w = first_frame_latent.shape
    ratio = vae_temporal_compression
    num_frames = (num_latent_t - 1) * ratio + 1
    mask = torch.ones(b, 1, num_frames, h, w,
                      device=first_frame_latent.device, dtype=first_frame_latent.dtype)
    mask[:, :, 1:] = 0
    first = torch.repeat_interleave(mask[:, :, :1], dim=2, repeats=ratio)
    mask = torch.cat([first, mask[:, :, 1:]], dim=2)
    mask = mask.view(b, -1, ratio, h, w).transpose(1, 2)  # [B, ratio, num_latent_t, H, W]
    return torch.cat([mask, first_frame_latent], dim=1)


def flow_matching_noisy(clean: torch.Tensor, noise: torch.Tensor,
                         sigma: float, flow_shift: float = 6.0) -> tuple[torch.Tensor, float, float]:
    """(1-s)*x + s*n with shifted sigma. Returns noisy, shifted_sigma, timestep."""
    shifted = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
    noisy = (1.0 - shifted) * clean + shifted * noise
    return noisy, shifted, shifted * 1000.0


# ------------------------------------------------------------------ #
# Per-sample gradient snapshot
# ------------------------------------------------------------------ #
def snapshot_layer_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    param_map = dict(model.named_parameters())
    for name in TRACK_LAYERS:
        if name not in param_map:
            # Some models pack Q/K/V differently. Try nested attn1 name as fallback.
            alt = name.replace("blocks.0.to_q", "blocks.0.attn1.to_q") \
                       .replace("blocks.29.to_out", "blocks.29.attn1.to_out") \
                       .replace(".ffn.fc_in.", ".ffn.net.0.proj.")
            if alt in param_map:
                p = param_map[alt]
            else:
                print(f"WARN param not found: {name}")
                continue
        else:
            p = param_map[name]
        if p.grad is None:
            print(f"WARN no grad on {name}")
            continue
        out[name] = p.grad.detach().clone().float().cpu()
    return out


def run_one_model(model_name: str, model_dir: str, samples: list[Sample],
                    out_dir: Path, device: torch.device,
                    flow_shift: float = 6.0) -> dict:
    print(f"\n===== {model_name} =====")
    torch.cuda.empty_cache()
    model = build_model_from_diffusers(model_dir, device)
    print("model params:",
          sum(p.numel() for p in model.parameters()) / 1e9, "B")

    # Store per-sample grads (may be large: patch_embedding.weight = 1536*52*4 = 320K params).
    per_sample_grads: dict[str, list[torch.Tensor]] = {n: [] for n in TRACK_LAYERS}

    for i, s in enumerate(samples):
        model.zero_grad(set_to_none=True)

        # Move tensors to device / bf16 dtype
        vae_latent = s.vae_latent.unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1,16,T,H,W]
        first_frame_latent = s.first_frame_latent.unsqueeze(0).to(device, dtype=torch.bfloat16)
        clip_feature = s.clip_feature.unsqueeze(0).to(device, dtype=torch.bfloat16)
        text_embedding = s.text_embedding.unsqueeze(0).to(device, dtype=torch.bfloat16)
        # Note attention mask: use all ones over the raw text seq len
        text_mask = torch.ones(1, s.text_embedding.shape[0], device=device, dtype=torch.bfloat16)

        # Track tensors
        track_points = s.track_points.unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1, T_full, N, 2]
        track_visibility = s.track_visibility.unsqueeze(0).to(device, dtype=torch.bfloat16)

        # Truncate track T to (num_latent_t - 1) * ratio + 1
        num_latent_t = vae_latent.shape[2]
        expected_T = (num_latent_t - 1) * 4 + 1
        track_points = track_points[:, :expected_T]
        track_visibility = track_visibility[:, :expected_T]

        # Sample sigma ~ U(0,1)
        sigma_u = float(torch.rand(1).item())
        # Compute noise and noisy
        noise = torch.randn_like(vae_latent, dtype=torch.float32)
        clean_f = vae_latent.float()
        noisy_f, shifted_sigma, timestep_val = flow_matching_noisy(clean_f, noise, sigma_u,
                                                                    flow_shift=flow_shift)
        noisy = noisy_f.to(torch.bfloat16)  # [1, 16, T, H, W]

        # Build I2V concat
        cond20 = build_i2v_cond_concat(first_frame_latent, vae_temporal_compression=4)  # [1,20,T,H,W]
        hidden_states = torch.cat([noisy, cond20], dim=1)  # [1, 36, T, H, W]

        # Timestep tensor
        timestep = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16)

        # Forward + backward BOTH inside set_forward_context so grad-checkpoint
        # recompute finds the forward context.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
                set_forward_context(current_timestep=timestep, attn_metadata=None):
            pred = model(
                hidden_states=hidden_states,
                encoder_hidden_states=text_embedding,
                encoder_attention_mask=text_mask,
                timestep=timestep,
                encoder_hidden_states_image=clip_feature,
                track_points=track_points,
                track_visibility=track_visibility,
                return_dict=False,
            )
            # pred is [B, C, T, H, W]
            # Loss: MSE(pred, noise - clean) computed in fp32
            target = (noise - clean_f)  # [B,16,T,H,W]
            loss = F.mse_loss(pred.float(), target.float())
            loss.backward()

        # Snapshot
        gs = snapshot_layer_grads(model)
        for n, g in gs.items():
            per_sample_grads[n].append(g)

        if i < 3 or i % 8 == 0:
            print(f"  sample {i}: sigma_u={sigma_u:.3f} shifted={shifted_sigma:.3f} "
                  f"t={timestep_val:.1f} loss={loss.item():.4f} "
                  f"|g[patch_embed]|={gs.get('patch_embedding.proj.weight', torch.zeros(1)).norm().item():.4f}")
        # free bookkeeping
        del pred, hidden_states, noisy, noise, cond20, target, loss
        torch.cuda.empty_cache()

    # Save all grads
    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    torch.save({n: torch.stack(gs, dim=0) for n, gs in per_sample_grads.items()
                if len(gs) > 0},
               model_out / "grads.pt")

    # Compute stats
    stats: dict = {}
    for n in TRACK_LAYERS:
        if n not in per_sample_grads or not per_sample_grads[n]:
            continue
        g_stack = torch.stack(per_sample_grads[n], dim=0)  # [N, *param_shape]
        # Special split for patch_embedding
        if n == "patch_embedding.proj.weight":
            # Shape [N, out, in, kD, kH, kW] with in=52
            base_slice = g_stack[:, :, :36]  # [N, out, 36, ...]
            track_slice = g_stack[:, :, 36:]  # [N, out, 16, ...]
            for key, sub in [(f"{n}[:,:36]", base_slice), (f"{n}[:,36:]", track_slice)]:
                per_sample_norms = sub.flatten(1).norm(dim=1)  # [N]
                mean_grad = sub.mean(0)
                mean_grad_norm = mean_grad.flatten().norm().item()
                avg_per_sample_norm = per_sample_norms.mean().item()
                snr = mean_grad_norm / (avg_per_sample_norm + 1e-30)
                stats[key] = {
                    "avg_per_sample_norm": avg_per_sample_norm,
                    "mean_grad_norm": mean_grad_norm,
                    "snr": snr,
                    "n_params": int(sub[0].numel()),
                }
        else:
            per_sample_norms = g_stack.flatten(1).norm(dim=1)
            mean_grad = g_stack.mean(0)
            mean_grad_norm = mean_grad.flatten().norm().item()
            avg_per_sample_norm = per_sample_norms.mean().item()
            snr = mean_grad_norm / (avg_per_sample_norm + 1e-30)
            stats[n] = {
                "avg_per_sample_norm": avg_per_sample_norm,
                "mean_grad_norm": mean_grad_norm,
                "snr": snr,
                "n_params": int(g_stack[0].numel()),
            }

    # Free model
    del model
    torch.cuda.empty_cache()
    return stats


def print_table(all_stats: dict) -> None:
    print("\n\n===== SNR TABLE =====")
    print(f"{'model':<14} {'layer':<48} {'avg_per_sample':>16} {'mean_grad':>16} {'snr':>10}")
    print("-" * 110)
    for model, stats in all_stats.items():
        for layer, s in stats.items():
            print(f"{model:<14} {layer:<48} {s['avg_per_sample_norm']:>16.6f} "
                  f"{s['mean_grad_norm']:>16.6f} {s['snr']:>10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--parquet-glob", type=str,
                    default="/mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset/shard_*/**/*.parquet")
    ap.add_argument("--out-dir", type=str, default="/mnt/lustre/vlm-s4duan/gradient_analysis")
    ap.add_argument("--flow-shift", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default="", help="comma list of models to run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    samples = load_samples(args.parquet_glob, args.n_samples, seed=args.seed)
    assert len(samples) == args.n_samples, "not enough samples loaded"

    all_stats: dict = {}
    models_to_run = args.only.split(",") if args.only else list(MODELS.keys())
    for name in models_to_run:
        name = name.strip()
        if not name:
            continue
        stats = run_one_model(name, MODELS[name], samples, out_dir, device,
                              flow_shift=args.flow_shift)
        all_stats[name] = stats
        # Save partial after each model
        with open(out_dir / "summary.json", "w") as f:
            json.dump(all_stats, f, indent=2)

    print_table(all_stats)
    print(f"\nsummary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
