# SPDX-License-Identifier: Apache-2.0
"""Build a WanTrack init checkpoint from a base Wan T2V diffusers model.

The WanTrack DiT widens the patch-embed input to 52 channels
(16 noisy latent + 20 I2V cond + 16 track) and adds a ``track_encoder``. This
script produces a diffusers transformer dir whose weights load *strictly* into
``TrackWanTransformer3DModel``:
  - ``patch_embedding.weight`` zero-padded 16 -> 52 input channels (pretrained
    weights occupy the first 16; the new I2V + track channels start at zero, so a
    freshly converted model reproduces the teacher at step 0),
  - ``track_encoder.*`` added (proj zero-init -> zero track contribution at step 0),
  - ``config.json`` gets ``in_channels=52`` + ``track_config``.

Other pipeline components (vae / text_encoder / tokenizer / scheduler /
model_index.json) are symlinked so the output is a complete, loadable model dir.

Usage (no GPU / no fastvideo import needed):
    python data_pipeline/convert_trackwan_init.py \
        --base <base diffusers model dir> --out <output dir>
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
ID_DIM = 128
VAE_T_COMP = 4
TRACK_CONFIG = {
    "id_dim": ID_DIM,
    "track_channels": TRACK_CHANNELS,
    "vae_spatial_compression": 8,
    "vae_temporal_compression": VAE_T_COMP,
    "max_track_id": 100_000,
    "zero_init_head": True,
}


def build_track_encoder_state() -> dict[str, torch.Tensor]:
    """Match TrackEncoder's params: temporal_conv (default init) + proj (zero)."""
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base Wan T2V diffusers model dir (with transformer/).")
    p.add_argument("--out", required=True, help="Output model dir for the WanTrack init.")
    args = p.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    if not (base / "transformer" / "config.json").exists():
        raise FileNotFoundError(f"{base}/transformer/config.json not found")
    out.mkdir(parents=True, exist_ok=True)

    # 1) Symlink every top-level entry except transformer/ (vae, text_encoder, ...).
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

    # 2) Convert transformer/.
    tdir = out / "transformer"
    tdir.mkdir(exist_ok=True)
    cfg = json.loads((base / "transformer" / "config.json").read_text())
    base_in = int(cfg["in_channels"])
    cfg["in_channels"] = NEW_IN_CHANNELS
    cfg["track_config"] = TRACK_CONFIG
    (tdir / "config.json").write_text(json.dumps(cfg, indent=2))

    sf_files = sorted((base / "transformer").glob("*.safetensors"))
    if len(sf_files) != 1:
        raise NotImplementedError(f"Expected exactly 1 transformer safetensors, found {len(sf_files)}")
    state = load_file(str(sf_files[0]))

    pe_key = "patch_embedding.weight"
    if pe_key not in state:
        cands = [k for k in state if "patch_embed" in k and k.endswith(".weight")]
        if len(cands) != 1:
            raise KeyError(f"Could not find patch_embedding weight; candidates={cands}")
        pe_key = cands[0]
    w = state[pe_key]  # [out, base_in, 1, 2, 2]
    if w.shape[1] != base_in:
        raise ValueError(f"{pe_key} in_ch {w.shape[1]} != config in_channels {base_in}")
    new_w = torch.zeros((w.shape[0], NEW_IN_CHANNELS, *w.shape[2:]), dtype=w.dtype)
    new_w[:, :base_in] = w  # pretrained channels first; new channels zero
    state[pe_key] = new_w
    print(f"[convert] padded {pe_key}: {tuple(w.shape)} -> {tuple(new_w.shape)}")

    te_state = build_track_encoder_state()
    for k, v in te_state.items():
        state[k] = v.to(w.dtype)
    print(f"[convert] added {len(te_state)} track_encoder tensors")

    save_file(state, str(tdir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    print(f"[convert] wrote {tdir/'diffusion_pytorch_model.safetensors'} ({len(state)} keys)")
    print(f"[convert] done -> {out}")


if __name__ == "__main__":
    main()
