# SPDX-License-Identifier: Apache-2.0
"""Merge one released Pika H3 Flex adapter into its MiniMax-H3 trunk for QAD.

QAD needs the Flex teacher as plain weights: the student's fp32 masters start
from them and the NVFP4 export quantizes them. This writes a model directory
whose trunk (``transformer`` for FL2VA, ``transformer_ref`` for Ref2VA) holds
``W + (alpha / rank) * B @ A`` for every adapted module, merged in fp32 and
rounded once to the base dtype, so the result does not depend on the runtime
LoRA path. Every other component is a symlink to the base snapshot.

The adapter is located and validated with the inference loader
(``fastvideo.pipelines.basic.pika_h3_flex.adapter.load_adapter``); its sha256
must match the manifest. The output also carries ``pika_flex_qad.json``, the
sampling contract the QAD model profile reads.

    python scripts/distill/pika_h3_flex_qad/merge_pika_adapter.py \\
        --base <MiniMax-H3 snapshot @42ed227> --adapter Pika-Labs/Pika-H3-Flex \\
        --task ref2va --output data/models/pika-flex-ref2va-merged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fastvideo.configs.pipelines.pika_h3_flex import (PIKA_AUDIO_SCHEDULER_SHIFT, PIKA_DENOISING_STEPS,
                                                      PIKA_VIDEO_SCHEDULER_SHIFT)
from fastvideo.pipelines.basic.pika_h3_flex.adapter import load_adapter

CONTRACT_FILE = "pika_flex_qad.json"
CONTRACT_SCHEMA = "pika-flex-qad-v1"
_LORA_A = ".lora_A.default.weight"
_LORA_B = ".lora_B.default.weight"


def adapter_factors(path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``{module: (A [r, in], B [out, r])}`` from a PEFT adapter file."""
    factors: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
            for suffix, slot in ((_LORA_A, "A"), (_LORA_B, "B")):
                if key.endswith(suffix):
                    factors.setdefault(key[:-len(suffix)], {})[slot] = handle.get_tensor(key)
                    break
            else:
                raise ValueError(f"{path}: unexpected adapter tensor {key!r}")
    incomplete = sorted(module for module, pair in factors.items() if set(pair) != {"A", "B"})
    if incomplete:
        raise ValueError(f"{path}: modules without both LoRA factors: {incomplete[:5]}")
    return {module: (pair["A"], pair["B"]) for module, pair in factors.items()}


def merge_weight(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float) -> torch.Tensor:
    """``weight + scale * B @ A`` in fp32, rounded once to ``weight``'s dtype."""
    if lora_b.shape[1] != lora_a.shape[0] or (lora_b.shape[0], lora_a.shape[1]) != tuple(weight.shape):
        raise ValueError(f"LoRA shapes A {tuple(lora_a.shape)} / B {tuple(lora_b.shape)} do not fit weight "
                         f"{tuple(weight.shape)}")
    delta = lora_b.to(torch.float32) @ lora_a.to(torch.float32)
    return (weight.to(torch.float32) + scale * delta).to(weight.dtype)


def merge_trunk(trunk_dir: Path, output_dir: Path, factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
                scale: float) -> dict[str, Any]:
    """Write ``trunk_dir``'s shards with every adapted module merged; return a per-module report."""
    index_path = trunk_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(factors)
    report: dict[str, float] = {}
    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(trunk_dir / shard), framework="pt") as handle:
            metadata = handle.metadata()
            for key in handle.keys():  # noqa: SIM118
                tensor = handle.get_tensor(key)
                module = key[:-len(".weight")] if key.endswith(".weight") else None
                if module in factors:
                    lora_a, lora_b = factors[module]
                    merged = merge_weight(tensor, lora_a, lora_b, scale)
                    report[module] = float((merged.float() - tensor.float()).norm() / tensor.float().norm())
                    remaining.discard(module)
                    tensor = merged
                tensors[key] = tensor.contiguous()
        save_file(tensors, str(output_dir / shard), metadata=metadata)
    if remaining:
        raise ValueError(f"{len(remaining)} adapter modules have no weight in {trunk_dir}: "
                         f"{sorted(remaining)[:5]}")
    (output_dir / index_path.name).write_text(index_path.read_text())
    (output_dir / "config.json").write_text((trunk_dir / "config.json").read_text())
    return {"modules": len(report), "max_rel_delta": max(report.values()), "rel_delta": report}


def link_base(base: Path, output: Path, trunk: str) -> None:
    """Symlink every base entry except the merged trunk."""
    output.mkdir(parents=True, exist_ok=True)
    for entry in sorted(base.iterdir()):
        if entry.name == trunk:
            continue
        target = output / entry.name
        if target.is_symlink() or target.exists():
            raise FileExistsError(f"{target} already exists; use an empty output directory")
        os.symlink(entry.resolve(), target)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="MiniMax-H3 snapshot at the adapter's pinned revision")
    parser.add_argument("--adapter", required=True, help="adapter file, release directory, or Hub repo id")
    parser.add_argument("--task", required=True, choices=("fl2va", "ref2va"))
    parser.add_argument("--revision", default=None, help="Hub revision when --adapter is a repo id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = load_adapter(args.adapter, task=args.task, revision=args.revision)
    digest = sha256_file(adapter.path)
    if adapter.manifest.get("sha256") and digest != adapter.manifest["sha256"]:
        raise ValueError(f"{adapter.path} has sha256 {digest}, the manifest says {adapter.manifest['sha256']}")
    base, output = Path(args.base).resolve(), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    factors = adapter_factors(adapter.path)
    scale = float(adapter.manifest["alpha"]) / float(adapter.manifest["rank"])
    link_base(base, output, adapter.trunk)
    report = merge_trunk(base / adapter.trunk, output / adapter.trunk, factors, scale)
    contract = {
        "schema": CONTRACT_SCHEMA,
        "task": args.task,
        "trunk": adapter.trunk,
        "release_id": adapter.manifest.get("release_id"),
        "adapter_sha256": digest,
        "base": adapter.manifest.get("base"),
        "numerics_policy": adapter.manifest.get("numerics_policy"),
        "denoising_steps": list(PIKA_DENOISING_STEPS),
        "video_scheduler_shift": PIKA_VIDEO_SCHEDULER_SHIFT,
        "audio_scheduler_shift": PIKA_AUDIO_SCHEDULER_SHIFT,
        "sampler": "renoise",
        "merged_modules": report["modules"],
        "merge": "fp32 W + (alpha/rank) B@A, rounded once to the base dtype",
    }
    (output / CONTRACT_FILE).write_text(json.dumps(contract, indent=2) + "\n")
    (output / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"merged {report['modules']} modules into {output / adapter.trunk} "
          f"(max relative weight change {report['max_rel_delta']:.4f})")


if __name__ == "__main__":
    main()
