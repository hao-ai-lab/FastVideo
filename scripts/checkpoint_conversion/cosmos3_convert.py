#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a Diffusers-format Cosmos3 checkpoint to FastVideo state-dict layout.

Cosmos3 weights (when published as ``nvidia/Cosmos3-Nano``) ship in a
Diffusers-style namespace with ``model.embed_tokens`` /
``model.layers.{i}.self_attn.*_moe_gen`` / ``vae2llm`` / ``llm2vae`` /
``time_embedder`` layout. FastVideo's ``Cosmos3VFMTransformer`` splits the
UND/GEN paths into separate ``language_model.layers.{i}`` and
``gen_layers.{i}`` subtrees. This script applies the 14-rule remap from
``Cosmos3OmniDiffusersPipeline._remap_ckpt_key`` to produce a state-dict
that loads cleanly into
``fastvideo.models.dits.cosmos3.Cosmos3VFMTransformer``.

Reference: vllm-omni PR #3454 (https://github.com/vllm-project/vllm-omni/pull/3454),
HEAD ``8536f5b1421f``; upstream ``pipeline_cosmos3.py:319-411``.

Usage:
    python scripts/checkpoint_conversion/cosmos3_convert.py \\
        --src /path/to/nvidia--Cosmos3-Nano \\
        --dst /path/to/cosmos3_nano_fastvideo
    python scripts/checkpoint_conversion/cosmos3_convert.py --smoke-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import safetensors.torch
import torch

from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
    Cosmos3OmniDiffusersPipeline,
)


def convert_state_dict(
    src_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Apply the Cosmos3 remap to a single state dict.

    Returns ``(new_state, skipped_keys, unmapped_keys)`` where:

    * ``new_state`` is the FastVideo-namespaced state dict.
    * ``skipped_keys`` lists source keys deliberately dropped by the remap
      (currently only ``lm_head.*``; the language-model head is not part of
      the diffusion transformer).
    * ``unmapped_keys`` lists source keys that did not match any remap rule.
      Should be empty for a clean Diffusers Cosmos3 checkpoint; non-empty
      results signal either an unexpected upstream key or a missing rule.
    """
    new_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unmapped: list[str] = []
    for src_key, tensor in src_state.items():
        new_key = Cosmos3OmniDiffusersPipeline._remap_ckpt_key(src_key)
        if new_key is None:
            # Either a deliberate skip (lm_head.*) or an unmapped key.
            if src_key.startswith("lm_head.") or src_key.startswith(
                    "transformer.lm_head."):
                skipped.append(src_key)
            else:
                unmapped.append(src_key)
            continue
        if new_key in new_state:
            raise KeyError(
                f"Duplicate target key {new_key!r} produced by remap; "
                f"colliding source keys include {src_key!r}")
        new_state[new_key] = tensor
    return new_state, skipped, unmapped


def convert_checkpoint_dir(src: Path, dst: Path) -> None:
    """Convert all ``*.safetensors`` shards in ``src`` and write to ``dst``.

    Reads each safetensors shard, applies the remap, and writes a single
    consolidated safetensors file at ``dst/model.safetensors``. This keeps
    the converter dependency surface small (single file out) for early
    Phase 5 work; once shard splitting becomes necessary, switch to
    per-shard outputs and emit ``model.safetensors.index.json``.

    TODO: Emit a sharded layout with ``model.safetensors.index.json`` when
    the real ``nvidia/Cosmos3-Nano`` weights land and exceed the
    single-file safetensors size budget.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint directory not found: {src}")
    if not src.is_dir():
        raise NotADirectoryError(
            f"Source path is not a directory: {src}")

    shard_paths = sorted(src.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No *.safetensors files found in {src}. Expected a Diffusers-format "
            "checkpoint directory.")

    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(shard_paths)} shard(s) from {src}")
    src_state: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        print(f"  Reading {shard_path.name}")
        shard = safetensors.torch.load_file(str(shard_path))
        for key, tensor in shard.items():
            if key in src_state:
                raise KeyError(
                    f"Duplicate key {key!r} found across shards "
                    f"(latest in {shard_path.name}); aborting.")
            src_state[key] = tensor

    print(f"Loaded {len(src_state)} parameters; applying remap")
    new_state, skipped, unmapped = convert_state_dict(src_state)
    print(f"  Remapped {len(new_state)} parameters")
    print(f"  Skipped  {len(skipped)} parameters (lm_head.*)")
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped key(s):")
        for k in unmapped[:10]:
            print(f"    {k}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")

    out_path = dst / "model.safetensors"
    print(f"Writing {out_path}")
    safetensors.torch.save_file(new_state, str(out_path))
    size_gb = out_path.stat().st_size / (1024**3)
    print(f"  Wrote {size_gb:.2f} GB to {out_path}")


def smoke_test() -> int:
    """Exercise the remap on a synthetic state dict covering all rule branches.

    Runs WITHOUT real weights and acts as a CI guard against drift between
    ``_remap_ckpt_key`` and this converter. Returns 0 on success, 1 on
    failure.

    The synthetic input contains one representative key per remap branch:
    top-level (``vae2llm`` / ``llm2vae`` / ``time_embedder``), language-model
    embeddings/norms, the standalone ``norm_moe_gen``, both UND and GEN
    attention/norm/MLP variants inside ``model.layers.{i}.*``, and the
    deliberately-skipped ``lm_head.weight``.
    """
    expected: dict[str, str | None] = {
        # Language-model trunk (UND path).
        "model.embed_tokens.weight":
            "transformer.language_model.embed_tokens.weight",
        "model.norm.weight":
            "transformer.language_model.norm.weight",
        # Standalone GEN norm at the top of the trunk.
        "model.norm_moe_gen.weight": "transformer.norm_moe_gen.weight",
        # UND self-attention (representative: q_proj).
        "model.layers.3.self_attn.q_proj.weight":
            "transformer.language_model.layers.3.self_attn.q_proj.weight",
        # GEN cross-attention via *_moe_gen suffix (representative: q_proj).
        "model.layers.3.self_attn.q_proj_moe_gen.weight":
            "transformer.gen_layers.3.cross_attention.q_proj.weight",
        # GEN cross-attention norm (representative: k_norm_moe_gen).
        "model.layers.3.self_attn.k_norm_moe_gen.weight":
            "transformer.gen_layers.3.cross_attention.k_norm.weight",
        # UND block norms.
        "model.layers.3.input_layernorm.weight":
            "transformer.language_model.layers.3.input_layernorm.weight",
        # GEN block norms.
        "model.layers.3.input_layernorm_moe_gen.weight":
            "transformer.gen_layers.3.input_layernorm.weight",
        # UND MLP (representative: gate_proj).
        "model.layers.3.mlp.gate_proj.weight":
            "transformer.language_model.layers.3.mlp.gate_proj.weight",
        # GEN MLP (representative: up_proj).
        "model.layers.3.mlp_moe_gen.up_proj.weight":
            "transformer.gen_layers.3.mlp.up_proj.weight",
        # Top-level generation projections.
        "vae2llm.weight": "transformer.vae2llm.weight",
        "llm2vae.weight": "transformer.llm2vae.weight",
        "time_embedder.linear_1.weight":
            "transformer.time_embedder.linear_1.weight",
        # Deliberately skipped.
        "lm_head.weight": None,
    }

    synthetic = {k: torch.zeros(1) for k in expected}
    new_state, skipped, unmapped = convert_state_dict(synthetic)

    failures: list[str] = []
    for src_key, expected_tgt in expected.items():
        if expected_tgt is None:
            if src_key not in skipped:
                failures.append(
                    f"  expected {src_key!r} in skipped; "
                    f"new_state contains: {sorted(new_state)}; "
                    f"unmapped contains: {unmapped}")
            continue
        if expected_tgt not in new_state:
            failures.append(
                f"  expected target key {expected_tgt!r} (from src "
                f"{src_key!r}) in new_state; got keys {sorted(new_state)}")
    if unmapped:
        failures.append(
            f"  unmapped keys (should be empty): {unmapped}")

    if failures:
        print("smoke_test FAILED:")
        for line in failures:
            print(line)
        return 1
    print(
        f"smoke_test PASSED: {len(new_state)} new keys, "
        f"{len(skipped)} skipped, {len(unmapped)} unmapped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Diffusers-format Cosmos3 checkpoint to "
                    "FastVideo state-dict layout.")
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Diffusers-format source checkpoint directory (e.g. an HF "
             "snapshot of nvidia/Cosmos3-Nano).")
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output directory; the converted state dict is written to "
             "<dst>/model.safetensors.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a no-weights smoke test that exercises every remap branch "
             "on a synthetic state dict. Useful as a CI guard.")
    args = parser.parse_args()

    if args.smoke_test:
        return smoke_test()
    if args.src is None or args.dst is None:
        parser.error("--src and --dst are required when not running --smoke-test")
    convert_checkpoint_dir(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
