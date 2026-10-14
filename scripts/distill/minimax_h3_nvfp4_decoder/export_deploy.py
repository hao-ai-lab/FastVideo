# SPDX-License-Identifier: Apache-2.0
"""Export a ``train_qad.py`` checkpoint to an inference-only file.

Drops the optimizer state and stores each ``NVFP4DecoderLinear`` master weight
and bias in bf16: both the training forward and inference quantize from the
bf16 cast (``weight.to(compute_dtype)``), so the packed NVFP4 weights, and
therefore every decoded pixel, are unchanged. All other tensors (norms,
scales, register tokens, ``proj_in``/``proj_out``, ``input_amax``) keep their
dtype. The run's ``metadata.json`` (student layers, act scale, rotation) is
embedded so the file is self-describing.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

# Same pattern as minimax_h3_nvfp4_decoder.DECODER_BLOCK_LINEAR; inlined so the export runs on GPU-less hosts
# (importing fastvideo initializes Triton, which needs a driver).
DECODER_BLOCK_LINEAR = re.compile(r"^transformer_blocks\.(\d+)\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0"
                                  r"|ff\.net\.0\.proj|ff\.net\.2)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="train_qad.py best.pt / last.pt")
    parser.add_argument("output", help="destination .pt")
    parser.add_argument("--metadata", help="run metadata.json (default: next to the checkpoint)")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    metadata_path = Path(args.metadata or Path(args.checkpoint).with_name("metadata.json"))
    metadata = {**state.get("metadata", {}), **json.loads(metadata_path.read_text())}
    decoder = {}
    for key, value in state["decoder"].items():
        module, _, param = key.rpartition(".")
        nvfp4 = DECODER_BLOCK_LINEAR.match(re.sub(r"\.base_layer$", "", module)) is not None
        decoder[key] = value.to(torch.bfloat16) if nvfp4 and param in ("weight", "bias") else value.clone()
    exported = {
        "format": "fastvideo_h3_decoder_nvfp4_deploy_v1",
        "decoder": decoder,
        "post_quant_conv": {k: v.clone() for k, v in state["post_quant_conv"].items()},
        "metadata": {
            **metadata, "source_step": state.get("step")
        },
    }
    torch.save(exported, args.output)
    size = Path(args.output).stat().st_size / 2**30
    print(json.dumps({"output": args.output, "gib": round(size, 2), "metadata": exported["metadata"]}))


if __name__ == "__main__":
    main()
