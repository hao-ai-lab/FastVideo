# SPDX-License-Identifier: Apache-2.0
"""Export a pre-packed NVFP4 decoder (``fastvideo_h3_decoder_nvfp4_packed_v1`` safetensors).

Loads an exported bf16-master decoder (``export_deploy.py`` .pt or its safetensors) into a dense
H3 VAE exactly as the pipeline loader does, freezes it on a Blackwell GPU, and writes the frozen
layers' packed tensors. The loader then skips the bf16 masters entirely.

    python export_packed.py decoder.safetensors /path/to/MiniMax-H3/vae decoder-nvfp4.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "minimax_h3_vae"))
from bench_decoder import load_h3_vae  # noqa: E402

from fastvideo.models.vaes.minimax_h3_nvfp4_checkpoint import (  # noqa: E402
    packed_nvfp4_decoder_checkpoint, write_nvfp4_decoder_safetensors)
from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (  # noqa: E402
    apply_nvfp4_decoder_checkpoint, load_nvfp4_decoder_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decoder", help="bf16-master decoder (.pt from export_deploy.py, or .safetensors)")
    parser.add_argument("host_vae_dir", help="MiniMax-H3 vae/ folder with at least the decoder's block count")
    parser.add_argument("output", help="destination .safetensors")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("packing needs a CUDA Blackwell GPU (FlashInfer FP4 quantization)")

    device = torch.device("cuda", torch.cuda.current_device())
    vae = load_h3_vae(args.host_vae_dir, device)
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
