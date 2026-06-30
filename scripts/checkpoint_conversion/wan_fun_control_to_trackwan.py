# SPDX-License-Identifier: Apache-2.0
"""Build a WanTrack init from the official Wan2.1-Fun-1.3B-Control (VideoX-Fun format).

MotionStream-style init. Instead of zero-padding fresh track channels onto a Fun-InP
base (which trains the track injection FROM SCRATCH and -- with a naive double zero-init --
deadlocks), we inherit Fun-Control's *pretrained* control channels, which already know how
to route a dense spatial control latent into the diffusion. This is exactly what MotionStream
does ("initialize from VideoX-Fun's Wan variants which extend Wan I2V with additional control
channels ... accelerates convergence").

Fun-Control's DiT input is 48ch, assembled by the VideoX-Fun pipeline as
    x = cat([ noisy(16) | control(16) | start_image/ref(16) ])
We remap those pretrained input channels into our EXISTING 52ch TrackWan layout so the model
definition is untouched (the wrapper builds [noisy|mask|first-frame], the transformer appends
the track map):
    our [0:16]   noisy        <- control [0:16]
    our [16:20]  mask         <- 0      (Fun-Control has no mask; the wrapper still builds one,
                                         projected by these zero weights -> ignored at init,
                                         trainable thereafter)
    our [20:36]  first-frame  <- control [32:48]   (ref / start-image slot)
    our [36:52]  track        <- control [16:32]   (PRETRAINED control channels -> warm start)

The track head ``proj`` is zero-init (``zero_init_head=True``): step-0 track map cm=0 -> the
track slot contributes nothing -> the model reproduces Fun-Control-as-I2V (teacher) exactly.
Gradient still flows because the track-slot patch-embed weights are Control's pretrained
control channels (grad(proj) ~ those weights != 0) -- so NO deadlock, unlike the from-scratch
Fun-InP path.

The source DiT uses VideoX-Fun naming; it is renamed to diffusers naming with the same map as
``scripts/checkpoint_conversion/wan_to_diffusers.py`` so the result loads *strictly* into
``TrackWanTransformer3DModel``. Non-transformer components (vae / text_encoder / tokenizer /
image_encoder / scheduler) and the diffusers arch config are taken from a Fun-InP *diffusers*
base -- identical 1.3B arch, same VAE / T5 / CLIP / scheduler. The converted body keys are
validated against that base's key set so a missing/extra key fails loudly here, not at train.

Usage (no GPU / no fastvideo import needed):
    python scripts/checkpoint_conversion/wan_fun_control_to_trackwan.py \
        --inp-base <Fun-InP diffusers dir> \
        --control-ckpt <alibaba-pai Fun-Control diffusion_pytorch_model.safetensors> \
        --out <output dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

NEW_IN_CHANNELS = 52
TRACK_CHANNELS = 16
ID_DIM = 128
VAE_T_COMP = 4
TRACK_CONFIG = {
    "id_dim": ID_DIM,
    "track_channels": TRACK_CHANNELS,
    "vae_spatial_compression": 8,
    "vae_temporal_compression": VAE_T_COMP,
    "max_track_id": 100_000,
    # Safe to zero-init the track head here: the track-slot patch-embed channels are
    # Fun-Control's PRETRAINED control channels (non-zero), so gradient reaches proj even
    # with cm=0 at step 0. zero-init proj => step-0 == teacher (Fun-Control as pure I2V).
    "zero_init_head": True,
}

# VideoX-Fun (alibaba-pai) -> diffusers WanTransformer3DModel naming.
# Copied from scripts/checkpoint_conversion/wan_to_diffusers.py (that module runs conversion
# code at import time, so it cannot be imported). ``patch_embedding`` keeps its name and is
# handled separately (channel remap), so it is intentionally absent here.
_PARAM_NAMES_MAPPING: dict = {
    r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.linear_1.\1",
    r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.linear_2.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.linear_1.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.linear_2.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_proj.\1",
    r"^img_emb\.proj\.0\.(.*)$": r"condition_embedder.image_embedder.norm1.\1",
    r"^img_emb\.proj\.1\.(.*)$": r"condition_embedder.image_embedder.ff.net.0.proj.\1",
    r"^img_emb\.proj\.3\.(.*)$": r"condition_embedder.image_embedder.ff.net.2.\1",
    r"^img_emb\.proj\.4\.(.*)$": r"condition_embedder.image_embedder.norm2.\1",
    r"^head\.modulation": r"scale_shift_table",
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.0.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.attn1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.attn1.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$": r"blocks.\1.attn2.add_k_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$": r"blocks.\1.attn2.add_v_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.0.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$": r"blocks.\1.attn2.norm_added_k.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.net.0.proj.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.net.2.\2",
    r"^blocks\.(\d+)\.modulation": r"blocks.\1.scale_shift_table",
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.norm2.\2",
}


def convert_names(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename VideoX-Fun keys -> diffusers keys. patch_embedding passes through verbatim."""
    out: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        if k.startswith("patch_embedding."):
            out[k] = v
            continue
        for pat, repl in _PARAM_NAMES_MAPPING.items():
            if re.match(pat, k):
                out[re.sub(pat, repl, k)] = v
                break
        else:
            raise ValueError(f"No diffusers-name mapping for source key: {k}")
    return out


def build_track_encoder_state() -> dict[str, torch.Tensor]:
    """temporal_conv (default init) + proj (ZERO). Zero proj is safe here -- see TRACK_CONFIG."""
    temporal_conv = nn.Conv3d(ID_DIM, TRACK_CHANNELS, kernel_size=(VAE_T_COMP, 1, 1), stride=(VAE_T_COMP, 1, 1))
    proj = nn.Conv3d(TRACK_CHANNELS, TRACK_CHANNELS, kernel_size=1)
    nn.init.zeros_(proj.weight)
    nn.init.zeros_(proj.bias)
    return {
        "track_encoder.temporal_conv.weight": temporal_conv.weight.detach().clone(),
        "track_encoder.temporal_conv.bias": temporal_conv.bias.detach().clone(),
        "track_encoder.proj.weight": proj.weight.detach().clone(),
        "track_encoder.proj.bias": proj.bias.detach().clone(),
    }


def remap_patch_embedding(ctrl_pe: torch.Tensor) -> torch.Tensor:
    """Fun-Control patch-embed [out,48,1,2,2] -> our [out,52,1,2,2] layout.

    Control order  : [ noisy 0:16 | control 16:32 | ref 32:48 ]
    Our (52) order : [ noisy 0:16 | mask 16:20 | first-frame 20:36 | track 36:52 ]
    """
    if ctrl_pe.shape[1] != 48:
        raise ValueError(f"expected Fun-Control patch_embedding in_ch=48, got {ctrl_pe.shape[1]}")
    out = torch.zeros((ctrl_pe.shape[0], NEW_IN_CHANNELS, *ctrl_pe.shape[2:]), dtype=ctrl_pe.dtype)
    out[:, 0:16] = ctrl_pe[:, 0:16]    # noisy            <- control noisy
    # out[:, 16:20] mask -> stays zero (Fun-Control has no mask channels)
    out[:, 20:36] = ctrl_pe[:, 32:48]  # first-frame      <- control ref / start-image
    out[:, 36:52] = ctrl_pe[:, 16:32]  # track (pretrained) <- control control-slot
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inp-base", required=True,
                   help="Fun-InP *diffusers* dir (config template + vae/text/clip/scheduler to symlink).")
    p.add_argument("--control-ckpt", required=True,
                   help="alibaba-pai Wan2.1-Fun-1.3B-Control diffusion_pytorch_model.safetensors (VideoX-Fun fmt).")
    p.add_argument("--out", required=True, help="Output WanTrack(Control) init dir.")
    args = p.parse_args()

    base = Path(args.inp_base)
    out = Path(args.out)
    if not (base / "transformer" / "config.json").exists():
        raise FileNotFoundError(f"{base}/transformer/config.json not found")
    out.mkdir(parents=True, exist_ok=True)

    # 1) Reference key set from the Fun-InP diffusers base (guarantees a strict load).
    base_sf = sorted((base / "transformer").glob("*.safetensors"))
    if not base_sf:
        raise FileNotFoundError(f"No transformer safetensors under {base}/transformer")
    base_state: dict[str, torch.Tensor] = {}
    for sf in base_sf:
        base_state.update(load_file(str(sf)))
    ref_keys = set(base_state)
    has_norm_added_q = any(k.endswith("attn2.norm_added_q.weight") for k in ref_keys)

    # 2) Load + rename the Fun-Control DiT.
    ctrl_src = load_file(args.control_ckpt)
    conv = convert_names(ctrl_src)

    # diffusers WanAttention pairs norm_added_k with norm_added_q; VideoX-Fun has no norm_q_img,
    # so synthesize zeros to match the base key set (only if the base actually has it).
    if has_norm_added_q:
        for k, v in list(conv.items()):
            if k.endswith("attn2.norm_added_k.weight"):
                conv[k.replace("norm_added_k", "norm_added_q")] = torch.zeros_like(v)

    # 3) Remap patch-embed 48 -> 52 (our layout); bias is per-output-channel -> unchanged.
    conv["patch_embedding.weight"] = remap_patch_embedding(conv["patch_embedding.weight"])

    # 4) Validate body keys == base body keys (ignore patch_embedding shape; track_encoder is new).
    conv_body = {k for k in conv}
    missing = ref_keys - conv_body
    extra = conv_body - ref_keys
    if missing or extra:
        raise RuntimeError(f"converted body keys != base keys.\n  missing={sorted(missing)[:8]}\n  "
                           f"extra={sorted(extra)[:8]}\n  (#missing={len(missing)} #extra={len(extra)})")
    for k in ref_keys:
        if k == "patch_embedding.weight":
            continue
        if conv[k].shape != base_state[k].shape:
            raise RuntimeError(f"shape mismatch {k}: control {tuple(conv[k].shape)} vs base {tuple(base_state[k].shape)}")
    print(f"[convert] body keys validated against base ({len(ref_keys)} keys); norm_added_q={'yes' if has_norm_added_q else 'no'}")

    # 5) Add the track encoder.
    dtype = conv["patch_embedding.weight"].dtype
    for k, v in build_track_encoder_state().items():
        conv[k] = v.to(dtype)
    print(f"[convert] patch_embedding -> {tuple(conv['patch_embedding.weight'].shape)}; +4 track_encoder tensors")

    # 6) Symlink every base entry except transformer/.
    for entry in os.listdir(base):
        if entry == "transformer":
            continue
        src = (base / entry).resolve()
        dst = out / entry
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        os.symlink(src, dst)

    # 7) Write transformer config (Fun-InP diffusers arch == Fun-Control arch) + our fields.
    tdir = out / "transformer"
    tdir.mkdir(exist_ok=True)
    cfg = json.loads((base / "transformer" / "config.json").read_text())
    cfg["in_channels"] = NEW_IN_CHANNELS
    cfg["out_channels"] = 16
    cfg["track_config"] = TRACK_CONFIG
    (tdir / "config.json").write_text(json.dumps(cfg, indent=2))

    save_file(conv, str(tdir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    print(f"[convert] wrote {tdir/'diffusion_pytorch_model.safetensors'} ({len(conv)} keys)")
    print(f"[convert] done -> {out}")


if __name__ == "__main__":
    main()
