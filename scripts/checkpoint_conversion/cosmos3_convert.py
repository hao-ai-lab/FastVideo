#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 checkpoint strict-load verifier (no weight conversion required).

The published ``nvidia/Cosmos3-Nano`` checkpoint is diffusers-format and its
transformer weight keys map 1:1 (identity) onto FastVideo's native
``Cosmos3VFMTransformer`` parameters -- ``needs_conversion=no``. There is no
remap to apply; the checkpoint loads directly.

This utility verifies strict-load completeness (every checkpoint key has a
matching DiT parameter of the right shape, and every DiT parameter is provided
by the checkpoint) without allocating the full ~30 GB model, by reading
safetensors headers and instantiating the DiT on the ``meta`` device.

Usage:
    python scripts/checkpoint_conversion/cosmos3_convert.py \
        --transformer official_weights/cosmos3/transformer
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import torch
from safetensors import safe_open

from fastvideo.configs.models.dits.cosmos3 import Cosmos3VideoConfig
from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer


def checkpoint_key_shapes(transformer_dir: str) -> dict[str, tuple[int, ...]]:
    """Read ``{key: shape}`` from a sharded safetensors transformer dir."""
    shards = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors found in {transformer_dir}")
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def verify_strict_load(transformer_dir: str) -> None:
    """Raise SystemExit if the checkpoint does not strict-load into the DiT."""
    ckpt = checkpoint_key_shapes(transformer_dir)
    cfg = Cosmos3VideoConfig()
    with torch.device("meta"):
        dit = Cosmos3VFMTransformer(cfg, hf_config={})
    params = {name: tuple(p.shape) for name, p in dit.named_parameters()}
    buffers = {name for name, _ in dit.named_buffers()}
    name_map: dict[str, str] = cfg.arch_config.param_names_mapping

    def remap(key: str) -> str:
        for pattern, replacement in name_map.items():
            if re.match(pattern, key):
                return re.sub(pattern, replacement, key)
        return key

    mapped = {remap(key): shape for key, shape in ckpt.items()}
    unexpected = sorted(set(mapped) - set(params) - buffers)
    missing = sorted(set(params) - set(mapped))
    mismatched = [(k, mapped[k], params[k]) for k in (set(mapped) & set(params)) if mapped[k] != params[k]]

    if unexpected or missing or mismatched:
        raise SystemExit("strict-load FAILED: "
                         f"unexpected={unexpected[:10]} missing={missing[:10]} "
                         f"shape_mismatch={mismatched[:10]}")
    print(f"strict-load OK: {len(ckpt)} checkpoint keys map 1:1 onto "
          f"{len(params)} DiT params (identity; needs_conversion=no)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformer",
        default=os.path.join("official_weights", "cosmos3", "transformer"),
        help="path to the checkpoint transformer/ directory",
    )
    args = parser.parse_args()
    verify_strict_load(args.transformer)


if __name__ == "__main__":
    main()
