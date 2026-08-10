# SPDX-License-Identifier: Apache-2.0
"""Pre-quantized MLX checkpoint save/load for the FastWan DiT.

Loading the Diffusers fp32/fp16 checkpoint and quantizing at startup costs
both download size and load time on every run. This module persists an already
cast (and optionally already quantized) ``MLXWanDiT`` so 16 GB users download
and load roughly half the bytes and skip requantization entirely:

    dit = mlx_dit_from_diffusers_safetensors(ckpt, cfg, quantization="int8")
    save_mlx_dit_checkpoint(dit, "FastWan2.1-T2V-1.3B-mlx-int8")
    ...
    dit = load_mlx_dit_checkpoint("FastWan2.1-T2V-1.3B-mlx-int8")

Format (one directory):

- ``mlx_dit.safetensors`` — every array, saved with ``mx.save_safetensors``.
  Plain weights keep their key; a quantized weight ``K`` is stored as the
  packed ``K`` plus ``K.scales`` (and ``K.biases`` for affine modes).
- ``mlx_dit.json`` — format version, the model config, the quantization spec,
  and which keys are quantized, so the loader can rebuild ``QuantizedMatrix``
  objects without guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.fastwan import (
    MLXQuantizationSpec,
    MLXWanDiT,
    MLXWanTransformerBlock,
    QuantizedMatrix,
    ensure_quantization_supported,
)

logger = init_logger(__name__)

FORMAT_VERSION = 1
WEIGHTS_FILENAME = "mlx_dit.safetensors"
MANIFEST_FILENAME = "mlx_dit.json"

_BLOCK_PREFIX = "blocks"

_DTYPE_TO_NAME = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}


def _dtype_name(dtype) -> str:
    import mlx.core as mx

    for raw, name in _DTYPE_TO_NAME.items():
        if dtype == getattr(mx, raw):
            return name
    raise ValueError(f"Unsupported MLX dtype for checkpointing: {dtype}")


def _name_to_dtype(name: str):
    import mlx.core as mx

    return {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[name]


def _flatten_weights(dit: MLXWanDiT) -> dict[str, Any]:
    flat: dict[str, Any] = dict(dit.weights)
    for index, block in enumerate(dit.blocks):
        for name, value in block.weights.items():
            flat[f"{_BLOCK_PREFIX}.{index}.{name}"] = value
    return flat


def save_mlx_dit_checkpoint(dit: MLXWanDiT, checkpoint_dir: str | Path) -> Path:
    """Persist ``dit`` (plain or quantized) into ``checkpoint_dir``."""
    import mlx.core as mx

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {}
    quantized: dict[str, dict[str, Any]] = {}
    spec: MLXQuantizationSpec | None = None
    for key, value in _flatten_weights(dit).items():
        if isinstance(value, QuantizedMatrix):
            if spec is not None and value.spec != spec:
                raise ValueError(
                    f"Mixed quantization specs in one checkpoint ({spec} vs {value.spec} at '{key}') "
                    "are not supported.")
            spec = value.spec
            arrays[key] = value.weight
            arrays[f"{key}.scales"] = value.scales
            if value.biases is not None:
                arrays[f"{key}.biases"] = value.biases
            quantized[key] = {
                "dequantized_dtype": _dtype_name(value.dequantized_dtype),
                "has_biases": value.biases is not None,
            }
        else:
            arrays[key] = value

    manifest = {
        "format_version": FORMAT_VERSION,
        "config": dit.config,
        "num_blocks": len(dit.blocks),
        "quantization": None if spec is None else {
            "mode": spec.mode,
            "bits": spec.bits,
            "group_size": spec.group_size,
        },
        "quantized_keys": quantized,
    }

    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    mx.save_safetensors(str(weights_path), arrays)
    (checkpoint_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    logger.info("Saved MLX DiT checkpoint (%d arrays, quantization=%s) to %s",
                len(arrays), spec.label if spec else "none", checkpoint_dir)
    return checkpoint_dir


def load_mlx_dit_checkpoint(checkpoint_dir: str | Path, *, compile: bool = False) -> MLXWanDiT:
    """Rebuild an ``MLXWanDiT`` saved by :func:`save_mlx_dit_checkpoint`."""
    import mlx.core as mx

    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not manifest_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Not an MLX DiT checkpoint directory: {checkpoint_dir} "
            f"(expected {MANIFEST_FILENAME} and {WEIGHTS_FILENAME}).")

    manifest = json.loads(manifest_path.read_text())
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"MLX DiT checkpoint {checkpoint_dir} has format_version={version}; "
            f"this FastVideo build reads version {FORMAT_VERSION}. Re-export the checkpoint.")

    spec = None
    if manifest["quantization"] is not None:
        spec = MLXQuantizationSpec(**manifest["quantization"])
        # The packed layout of mx.quantize output is mode-specific, so a build
        # that cannot run the mode cannot use these arrays at all.
        ensure_quantization_supported(spec)

    arrays = mx.load(str(weights_path))
    quantized_keys: dict[str, dict[str, Any]] = manifest["quantized_keys"]

    def rebuild(key: str):
        if key not in quantized_keys:
            return arrays[key]
        info = quantized_keys[key]
        assert spec is not None, f"Quantized key '{key}' in a checkpoint without a quantization spec"
        return QuantizedMatrix(
            weight=arrays[key],
            scales=arrays[f"{key}.scales"],
            biases=arrays[f"{key}.biases"] if info["has_biases"] else None,
            spec=spec,
            dequantized_dtype=_name_to_dtype(info["dequantized_dtype"]),
        )

    config = manifest["config"]
    block_keys: dict[int, list[str]] = {}
    top_level_keys: list[str] = []
    for key in arrays:
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if key.startswith(f"{_BLOCK_PREFIX}."):
            index_str, _, _ = key[len(_BLOCK_PREFIX) + 1:].partition(".")
            block_keys.setdefault(int(index_str), []).append(key)
        else:
            top_level_keys.append(key)

    weights = {key: rebuild(key) for key in top_level_keys}

    num_blocks = int(manifest["num_blocks"])
    if sorted(block_keys) != list(range(num_blocks)):
        raise ValueError(
            f"MLX DiT checkpoint {checkpoint_dir} is missing block weights: "
            f"manifest says {num_blocks} blocks, found indices {sorted(block_keys)}.")

    inner_dim = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    blocks = []
    for index in range(num_blocks):
        prefix = f"{_BLOCK_PREFIX}.{index}."
        block_weights = {key[len(prefix):]: rebuild(key) for key in block_keys[index]}
        blocks.append(
            MLXWanTransformerBlock(
                block_weights,
                dim=inner_dim,
                ffn_dim=int(config["ffn_dim"]),
                num_heads=int(config["num_attention_heads"]),
                eps=float(config["eps"]),
            ))
    return MLXWanDiT(weights, blocks, config, compile=compile)
