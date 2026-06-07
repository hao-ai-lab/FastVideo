# SPDX-License-Identifier: Apache-2.0
"""Strict-load completeness: real Cosmos3-Nano transformer <-> FastVideo DiT.

The published ``nvidia/Cosmos3-Nano`` checkpoint is diffusers-format
(``needs_conversion=no``). This test verifies that EVERY transformer weight key
in the checkpoint maps 1:1 (via the DiT's ``param_names_mapping``) onto a
FastVideo ``Cosmos3VFMTransformer`` parameter of matching shape, and that no DiT
parameter is left unfilled -- i.e. a ``strict=True`` load will succeed.

It runs on the ``meta`` device (no 30 GB allocation, no GPU) by reading only the
safetensors headers. Skips cleanly if the checkpoint is not present locally.
"""
from __future__ import annotations

import glob
import os
import re

import pytest
import torch

pytestmark = [pytest.mark.local]

_CKPT_DIR = os.path.join("official_weights", "cosmos3", "transformer")


def _checkpoint_key_shapes() -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open
    shards = sorted(glob.glob(os.path.join(_CKPT_DIR, "*.safetensors")))
    if not shards:
        pytest.skip(f"Cosmos3 transformer checkpoint not present: {_CKPT_DIR}")
    out: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = tuple(f.get_slice(k).get_shape())
    return out


def _meta_dit():
    from fastvideo.configs.models.dits.cosmos3 import Cosmos3VideoConfig
    from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer
    cfg = Cosmos3VideoConfig()
    with torch.device("meta"):
        dit = Cosmos3VFMTransformer(cfg, hf_config={})
    return dit, cfg


def _apply_mapping(key: str, pmap: dict[str, str]) -> str:
    for pat, repl in pmap.items():
        if re.match(pat, key):
            return re.sub(pat, repl, key)
    return key


def test_strict_load_completeness():
    ckpt = _checkpoint_key_shapes()
    dit, cfg = _meta_dit()
    dit_params = {n: tuple(p.shape) for n, p in dit.named_parameters()}
    dit_buffers = {n: tuple(b.shape) for n, b in dit.named_buffers()}
    pmap = cfg.arch_config.param_names_mapping

    mapped = {_apply_mapping(k, pmap): v for k, v in ckpt.items()}
    ckpt_names = set(mapped)
    param_names = set(dit_params)
    buffer_names = set(dit_buffers)

    # Every checkpoint key must land on a DiT parameter (non-persistent buffers excepted).
    unexpected = sorted(ckpt_names - param_names - buffer_names)
    assert not unexpected, f"checkpoint keys with no DiT param: {unexpected[:20]}"

    # Every DiT parameter must be provided by the checkpoint (true strict load).
    missing = sorted(param_names - ckpt_names)
    assert not missing, f"DiT params not provided by checkpoint: {missing[:20]}"

    # Shapes must match exactly.
    mism = [(k, mapped[k], dit_params[k]) for k in (ckpt_names & param_names) if mapped[k] != dit_params[k]]
    assert not mism, f"shape mismatches: {mism[:10]}"

    assert len(ckpt) == len(dit_params), f"key count mismatch: ckpt={len(ckpt)} dit={len(dit_params)}"
