# SPDX-License-Identifier: Apache-2.0
"""Export a pre-packed NVFP4 decoder (``fastvideo_h3_decoder_nvfp4_packed_v1`` safetensors).

Loads an exported bf16-master decoder (``export_deploy.py`` .pt or its safetensors) into a dense
H3 VAE exactly as the pipeline loader does, freezes it on a Blackwell GPU, and writes the frozen
layers' packed tensors. The loader then skips the bf16 masters entirely.

    python export_packed.py decoder.safetensors /path/to/MiniMax-H3/vae decoder-nvfp4.safetensors
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from fastvideo.models.vaes.minimax_h3_nvfp4_checkpoint import (
    packed_nvfp4_decoder_checkpoint, write_nvfp4_decoder_safetensors)
from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (
    apply_nvfp4_decoder_checkpoint, load_nvfp4_decoder_checkpoint)


def _load_h3_vae(vae_dir: str, device: torch.device) -> torch.nn.Module:
    """The dense H3 VAE the decoder checkpoint was distilled against."""
    from safetensors.torch import load_file

    from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
    from fastvideo.models.vaes.minimax_h3_int8_convrot import dense_vae_safetensors
    from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3

    config = MiniMaxH3VideoVAEConfig()
    with open(os.path.join(vae_dir, "config.json")) as handle:
        config.update_model_arch(json.load(handle))
    vae = AutoencoderKLMiniMaxH3(config).to(device)
    state: dict[str, torch.Tensor] = {}
    for path in dense_vae_safetensors(sorted(glob.glob(os.path.join(vae_dir, "*.safetensors")))):
        state.update(load_file(path))
    vae.load_state_dict(state, strict=True)
    return vae.eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decoder", help="bf16-master decoder (.pt from export_deploy.py, or .safetensors)")
    parser.add_argument("host_vae_dir", help="MiniMax-H3 vae/ folder with at least the decoder's block count")
    parser.add_argument("output", help="destination .safetensors")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("packing needs a CUDA Blackwell GPU (FlashInfer FP4 quantization)")

    device = torch.device("cuda", torch.cuda.current_device())
    vae = _load_h3_vae(args.host_vae_dir, device)
    metadata = apply_nvfp4_decoder_checkpoint(vae, load_nvfp4_decoder_checkpoint(args.decoder), freeze=True)
    import flashinfer

    major, minor = torch.cuda.get_device_capability(device)
    packed_on = f"{torch.cuda.get_device_name(device)} sm_{major}{minor}"
    packed = packed_nvfp4_decoder_checkpoint(vae, {
        **metadata, "packed_on": packed_on,
        "flashinfer_version": flashinfer.__version__
    })
    write_nvfp4_decoder_safetensors(args.output, packed)
    print(json.dumps({"output": args.output, "bytes": os.path.getsize(args.output), "metadata": packed["metadata"]}))


if __name__ == "__main__":
    main()
