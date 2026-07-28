# SPDX-License-Identifier: Apache-2.0
"""Build a WanTrack init checkpoint from a base Wan diffusers model, with several
strategies for the patch_embedding track-channel slot and the track_encoder.

Modes for patch_embedding[:, base_in:] (the new 16 in-channels):
  --pe-init zero    : zero-pad (original stage-1 recipe)
  --pe-init random  : normal-init with std matched to base first-N channels

Modes for track_encoder.{proj, temporal_conv}:
  --track-src default            : default Conv init (used by original convert)
  --track-src <ckpt.safetensors> : lift both convs from an existing WanTrack safetensors
                                   (e.g. a trained 1.3B ckpt). Bias is copied when present
                                   and its shape matches.

Usage:
  python data_pipeline/convert_trackwan_init_v2.py \
      --base /mnt/lustre/vlm-s4duan/models/Wan2.1-I2V-14B-720P-Diffusers \
      --out  /mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_random_init \
      --id-dim 64 --pe-init random

  python data_pipeline/convert_trackwan_init_v2.py \
      --base /mnt/lustre/vlm-s4duan/models/Wan2.1-I2V-14B-720P-Diffusers \
      --out  /mnt/lustre/vlm-s4duan/models/trackwan_14b_i2v_d64_partial_merged \
      --id-dim 64 --pe-init random \
      --track-src /mnt/lustre/vlm-s4duan/exports/synth_stage2_paperLR_ckpt600/transformer/model.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

NEW_IN_CHANNELS = 52
TRACK_CHANNELS = 16
VAE_T_COMP = 4
DEFAULT_ID_DIM = 128


def build_track_encoder_default(id_dim: int, dtype: torch.dtype, use_bias: bool = False) -> dict[str, torch.Tensor]:
    """Default (non-zero) Conv init for track_encoder.proj + temporal_conv."""
    temporal_conv = nn.Conv3d(id_dim, TRACK_CHANNELS,
                              kernel_size=(VAE_T_COMP, 1, 1), stride=(VAE_T_COMP, 1, 1),
                              bias=use_bias)
    proj = nn.Conv3d(TRACK_CHANNELS, TRACK_CHANNELS, kernel_size=1, bias=use_bias)
    d = {
        "track_encoder.temporal_conv.weight": temporal_conv.weight.detach().to(dtype).clone(),
        "track_encoder.proj.weight": proj.weight.detach().to(dtype).clone(),
    }
    if use_bias:
        d["track_encoder.temporal_conv.bias"] = temporal_conv.bias.detach().to(dtype).clone()
        d["track_encoder.proj.bias"] = proj.bias.detach().to(dtype).clone()
    return d


def load_src_state(src_path: str) -> dict[str, torch.Tensor]:
    """Load a source state dict from a .safetensors FILE or a diffusers model DIR.

    A 14B export is sharded across several *.safetensors, so accept a directory (either the
    model root or its transformer/ subdir) and merge every shard.
    """
    p = Path(src_path)
    if p.is_file():
        return load_file(str(p))
    tdir = p / "transformer" if (p / "transformer").is_dir() else p
    shards = sorted(tdir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {tdir}")
    state: dict[str, torch.Tensor] = {}
    for s in shards:
        state.update(load_file(str(s)))
    print(f"[src] loaded {len(state)} tensors from {len(shards)} shard(s) in {tdir}")
    return state


def lift_track_encoder_from_src(src_path: str, id_dim: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Load track_encoder.{proj, temporal_conv}.{weight, bias} from an existing WanTrack ckpt.

    Verifies shape compatibility. If bias is missing in the source we skip it; if the
    source shapes don't match the expected shapes we fall back to default init for that
    key and print a warning.
    """
    src_state = load_src_state(src_path)
    expected_shapes = {
        "track_encoder.temporal_conv.weight": (TRACK_CHANNELS, id_dim, VAE_T_COMP, 1, 1),
        "track_encoder.proj.weight":          (TRACK_CHANNELS, TRACK_CHANNELS, 1, 1, 1),
        "track_encoder.temporal_conv.bias":   (TRACK_CHANNELS,),
        "track_encoder.proj.bias":            (TRACK_CHANNELS,),
    }
    picked: dict[str, torch.Tensor] = {}
    for k, want in expected_shapes.items():
        if k not in src_state:
            continue
        got = tuple(src_state[k].shape)
        if got != want:
            print(f"[track-src] SHAPE MISMATCH on {k}: src={got} vs expected={want}; SKIPPING (will need fallback)")
            continue
        picked[k] = src_state[k].detach().to(dtype).clone()
        print(f"[track-src] lifted {k} shape={got}")
    # If either weight is missing, we still need to keep the model loadable.
    fallback = build_track_encoder_default(id_dim, dtype, use_bias=False)
    for k, v in fallback.items():
        if k not in picked:
            print(f"[track-src] {k} not in source -> default init")
            picked[k] = v
    return picked


def std_of_first_n(w: torch.Tensor, n: int) -> float:
    """Std of the pretrained first-N in-channels slice of patch_embedding.weight."""
    with torch.no_grad():
        s = w[:, :n].float().std().item()
    return float(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base Wan diffusers model dir.")
    p.add_argument("--out",  required=True, help="Output diffusers model dir.")
    p.add_argument("--id-dim", type=int, default=DEFAULT_ID_DIM)
    p.add_argument("--pe-init", choices=("zero", "random"), default="zero",
                   help="patch_embedding init strategy for the ADDED track channels [:, base_in:]. "
                        "Base pretrained channels [:, :base_in] are always preserved.")
    p.add_argument("--pe-random-seed", type=int, default=1234)
    p.add_argument("--track-src", default=None,
                   help="Path to a .safetensors (or diffusers model dir) with track_encoder.* to lift.")
    p.add_argument("--pe-src", default=None,
                   help="Path to a .safetensors (or diffusers model dir) to lift patch_embedding.weight[:, base_in:] "
                        "from — i.e. the TRACK SLOT only; the pretrained [:, :base_in] channels always come from "
                        "--base. Use together with --track-src pointing at the SAME source so the encoder and the "
                        "track slot stay CO-ADAPTED (lifting the encoder alone is worse than random init).")
    p.add_argument("--use-bias-defaults", action="store_true",
                   help="When --track-src is not set, build track_encoder convs WITH bias (matches TRACKWAN_TRACK_BIAS=1 training).")
    args = p.parse_args()

    id_dim = int(args.id_dim)
    TRACK_CONFIG = {
        "id_dim": id_dim,
        "track_channels": TRACK_CHANNELS,
        "vae_spatial_compression": 8,
        "vae_temporal_compression": VAE_T_COMP,
        "max_track_id": 100_000,
        "zero_init_head": False,
    }

    base = Path(args.base)
    out = Path(args.out)
    if not (base / "transformer" / "config.json").exists():
        raise FileNotFoundError(f"{base}/transformer/config.json not found")
    out.mkdir(parents=True, exist_ok=True)

    # 1) Symlink every top-level entry except transformer/
    for entry in os.listdir(base):
        if entry == "transformer":
            continue
        src_p = (base / entry).resolve()
        dst = out / entry
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        os.symlink(src_p, dst)

    # 2) Rewrite transformer/config.json
    tdir = out / "transformer"
    tdir.mkdir(exist_ok=True)
    cfg = json.loads((base / "transformer" / "config.json").read_text())
    base_in = int(cfg["in_channels"])
    cfg["in_channels"] = NEW_IN_CHANNELS
    cfg["track_config"] = TRACK_CONFIG
    (tdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # 3) Load full base transformer state
    sf_files = sorted((base / "transformer").glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors under {base}/transformer")
    state: dict[str, torch.Tensor] = {}
    for sf in sf_files:
        state.update(load_file(str(sf)))
    dtype = next(iter(state.values())).dtype

    # 4) Grow patch_embedding.weight from base_in -> 52 input channels
    pe_key = "patch_embedding.weight"
    if pe_key not in state:
        raise KeyError(f"{pe_key} not found in base transformer")
    w = state[pe_key]  # [hidden, base_in, 1, 2, 2]
    if w.shape[1] != base_in:
        raise ValueError(f"{pe_key} in_ch {w.shape[1]} != config in_channels {base_in}")
    new_w = torch.empty((w.shape[0], NEW_IN_CHANNELS, *w.shape[2:]), dtype=w.dtype)
    new_w[:, :base_in] = w  # pretrained channels first
    if args.pe_init == "zero":
        new_w[:, base_in:] = 0
        print(f"[convert] pe-init=zero -> new channels are zero")
    else:
        std = std_of_first_n(w, base_in)
        g = torch.Generator().manual_seed(int(args.pe_random_seed))
        noise = torch.empty_like(new_w[:, base_in:], dtype=torch.float32)
        noise.normal_(mean=0.0, std=std, generator=g)
        new_w[:, base_in:] = noise.to(w.dtype)
        print(f"[convert] pe-init=random N(0, {std:.5f}) matching first-{base_in} std")
    # 4b) Optionally overwrite ONLY the track slot [:, base_in:] from a trained source.
    # This is the other half of the "merged" recipe: the track slot and the track_encoder were
    # co-adapted during the overfit, so they must be lifted TOGETHER (--pe-src + --track-src from
    # the same ckpt). Lifting the encoder alone leaves it talking to a random projection.
    if args.pe_src:
        src_pe_state = load_src_state(args.pe_src)
        if pe_key not in src_pe_state:
            raise KeyError(f"{pe_key} not found in --pe-src {args.pe_src}")
        src_pe = src_pe_state[pe_key]
        want = (w.shape[0], NEW_IN_CHANNELS, *w.shape[2:])
        if tuple(src_pe.shape) != want:
            raise ValueError(f"--pe-src {pe_key} shape {tuple(src_pe.shape)} != expected {want}")
        slot = src_pe[:, base_in:].to(w.dtype)
        new_w[:, base_in:] = slot
        print(f"[pe-src] lifted {pe_key}[:, {base_in}:] from source "
              f"(std={slot.float().std().item():.5f}, absmax={slot.float().abs().max().item():.5f})")

    state[pe_key] = new_w
    print(f"[convert] patch_embedding.weight: {tuple(w.shape)} -> {tuple(new_w.shape)}")
    with torch.no_grad():
        print(f"[convert] pe[:, :{base_in}] std={new_w[:, :base_in].float().std().item():.5f} (pretrained) | "
              f"pe[:, {base_in}:] std={new_w[:, base_in:].float().std().item():.5f} (track slot)")

    # 5) Add track_encoder
    if args.track_src:
        te = lift_track_encoder_from_src(args.track_src, id_dim, dtype)
    else:
        te = build_track_encoder_default(id_dim, dtype, use_bias=args.use_bias_defaults)
    for k, v in te.items():
        state[k] = v
    print(f"[convert] added {len(te)} track_encoder tensors")

    save_file(state, str(tdir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    print(f"[convert] wrote {tdir/'diffusion_pytorch_model.safetensors'} ({len(state)} keys)")
    print(f"[convert] done -> {out}")


if __name__ == "__main__":
    main()
