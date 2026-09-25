# SPDX-License-Identifier: Apache-2.0
"""Convert Comfy-Org's MiniMax-H3 NVFP4-AWQ conditioner for FastVideo.

The public Comfy checkpoint already stores the 350 language linears as packed
E2M1 values with E4M3 block scales. This converter preserves those quantized
values instead of dequantizing and requantizing them. It changes only the
serialization contract FastVideo needs:

* rename the Comfy Qwen3-VL keys to FastVideo's native conditioner keys;
* swap each packed pair because Comfy stores the even value in the high nibble
  while FlashInfer consumes it in the low nibble;
* reinterpret the E4M3 scale bytes and invert Comfy's per-tensor scale;
* preserve AWQ ``pre_quant_scale`` vectors and add identity vectors to the
  other linears so the loader has one explicit contract; and
* dequantize the row-wise INT8 token embedding to bf16.

The output is a Hub-style sharded directory selected automatically through its
``config.json`` quantization metadata. No CUDA device is required because the
source is already quantized.

Usage::

    python scripts/checkpoint_conversion/convert_minimax_h3_comfy_nvfp4_awq.py \
        --src qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
        --dst /path/to/FastH3-comfy-nvfp4-awq/text_encoder
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from convert_minimax_h3_text_encoder_nvfp4 import ShardWriter
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLArchConfig
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import (
    LANGUAGE_PROJECTIONS,
    nvfp4_packed_weight_shape,
    nvfp4_scale_shape,
    serialized_nvfp4_quantization_config,
)

LANGUAGE_PREFIX = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.(?P<proj>" +
    "|".join(re.escape(name) for name in LANGUAGE_PROJECTIONS) + r")$")
COMFY_QUANT_SUFFIX = ".comfy_quant"
WEIGHT_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2")
EXPECTED_LAYERS = 50
ARCH = MiniMaxH3Qwen3VLArchConfig()
KV_SIZE = ARCH.num_key_value_heads * ARCH.head_dim
PROJECTION_SHAPES = {
    "self_attn.q_proj": (ARCH.hidden_size, ARCH.hidden_size),
    "self_attn.k_proj": (KV_SIZE, ARCH.hidden_size),
    "self_attn.v_proj": (KV_SIZE, ARCH.hidden_size),
    "self_attn.o_proj": (ARCH.hidden_size, ARCH.hidden_size),
    "mlp.gate_proj": (ARCH.intermediate_size, ARCH.hidden_size),
    "mlp.up_proj": (ARCH.intermediate_size, ARCH.hidden_size),
    "mlp.down_proj": (ARCH.hidden_size, ARCH.intermediate_size),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Comfy NVFP4-AWQ safetensors file")
    parser.add_argument("--dst", type=Path, required=True, help="text_encoder directory to write")
    parser.add_argument("--shard-size-gb", type=float, default=4.0, help="output safetensors shard size")
    parser.add_argument("--source-revision", help="optional source repository revision recorded in config.json")
    args = parser.parse_args()
    if args.shard_size_gb <= 0:
        parser.error("--shard-size-gb must be positive")
    return args


def fastvideo_name(name: str) -> str:
    if name.startswith("model.layers."):
        return "model.language_model.layers." + name[len("model.layers."):]
    if name.startswith("model.embed_tokens."):
        return "model.language_model.embed_tokens." + name[len("model.embed_tokens."):]
    if name.startswith("visual."):
        return "model." + name
    return name


def decode_marker(tensor: torch.Tensor, name: str) -> dict:
    try:
        marker = json.loads(bytes(tensor.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Comfy quantization marker {name}") from error
    if not isinstance(marker, dict) or not isinstance(marker.get("format"), str):
        raise ValueError(f"Comfy quantization marker {name} has no string format")
    return marker


def inspect_source(handle) -> tuple[set[str], set[str]]:
    keys = set(handle.keys())
    quantized: set[str] = set()
    for name in sorted(key for key in keys if key.endswith(COMFY_QUANT_SUFFIX)):
        prefix = name[:-len(COMFY_QUANT_SUFFIX)]
        marker = decode_marker(handle.get_tensor(name), name)
        if prefix == "model.embed_tokens":
            if marker["format"] != "int8_tensorwise":
                raise ValueError(f"Expected an int8_tensorwise token embedding, got {marker['format']!r}")
            continue
        match = LANGUAGE_PREFIX.match(prefix)
        if match is None or marker["format"] != "nvfp4":
            raise ValueError(f"Unsupported quantized tensor {prefix} with format {marker['format']!r}")
        quantized.add(prefix)

    expected = {
        f"model.layers.{layer}.{projection}"
        for layer in range(EXPECTED_LAYERS)
        for projection in LANGUAGE_PROJECTIONS
    }
    missing = sorted(expected - quantized)
    extra = sorted(quantized - expected)
    if missing or extra:
        raise ValueError(f"Expected {len(expected)} H3 language linears; missing {missing[:2]}, extra {extra[:2]}")
    for prefix in quantized:
        absent = [suffix for suffix in WEIGHT_SUFFIXES if prefix + suffix not in keys]
        if absent:
            raise ValueError(f"Quantized linear {prefix} is missing {absent}")
        projection = LANGUAGE_PREFIX.fullmatch(prefix)["proj"]
        output_size, input_size = PROJECTION_SHAPES[projection]
        expected_shapes = {
            ".weight": nvfp4_packed_weight_shape(output_size, input_size),
            ".weight_scale": nvfp4_scale_shape(output_size, input_size),
            ".weight_scale_2": (),
        }
        expected_dtypes = {
            ".weight": torch.uint8,
            ".weight_scale": torch.float8_e4m3fn,
            ".weight_scale_2": torch.float32,
        }
        for suffix in WEIGHT_SUFFIXES:
            tensor = handle.get_tensor(prefix + suffix)
            if tuple(tensor.shape) != expected_shapes[suffix] or tensor.dtype != expected_dtypes[suffix]:
                raise ValueError(
                    f"Unexpected {prefix + suffix} shape or dtype: got {tuple(tensor.shape)} {tensor.dtype}, "
                    f"expected {expected_shapes[suffix]} {expected_dtypes[suffix]}"
                )
    for name in ("model.embed_tokens.weight", "model.embed_tokens.weight_scale"):
        if name not in keys:
            raise ValueError(f"Comfy checkpoint is missing {name}")
    embedding = handle.get_tensor("model.embed_tokens.weight")
    embedding_scale = handle.get_tensor("model.embed_tokens.weight_scale")
    if embedding.shape != (ARCH.vocab_size, ARCH.hidden_size) or embedding.dtype != torch.int8:
        raise ValueError(f"Unexpected token embedding shape or dtype: {tuple(embedding.shape)} {embedding.dtype}")
    if embedding_scale.shape != (ARCH.vocab_size, 1) or embedding_scale.dtype != torch.float32:
        raise ValueError(
            f"Unexpected token embedding scale shape or dtype: {tuple(embedding_scale.shape)} {embedding_scale.dtype}"
        )
    pre_scaled = {name[:-len(".pre_quant_scale")] for name in keys if name.endswith(".pre_quant_scale")}
    unknown_pre_scales = sorted(pre_scaled - quantized)
    if unknown_pre_scales:
        raise ValueError(f"pre_quant_scale found on unquantized tensors {unknown_pre_scales[:2]}")
    for prefix in pre_scaled:
        projection = LANGUAGE_PREFIX.fullmatch(prefix)["proj"]
        pre_scale = handle.get_tensor(prefix + ".pre_quant_scale")
        input_size = PROJECTION_SHAPES[projection][1]
        if pre_scale.shape != (input_size, ) or pre_scale.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected {prefix}.pre_quant_scale shape or dtype")
    return quantized, pre_scaled


def staging_destination(dst: Path) -> Path:
    """Reserve a sibling directory so only a complete checkpoint becomes visible at ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not dst.is_dir() or any(dst.iterdir()):
            raise SystemExit(f"{dst} already exists and is not an empty directory; refusing to overwrite it")
        dst.rmdir()
    return Path(tempfile.mkdtemp(prefix=f".{dst.name}.partial-", dir=dst.parent))


def dequantize_embedding(weight: torch.Tensor, scale: torch.Tensor, rows: int = 4096) -> torch.Tensor:
    if weight.dtype != torch.int8 or scale.dtype != torch.float32:
        raise ValueError(f"Expected INT8 embedding and FP32 scale, got {weight.dtype} and {scale.dtype}")
    if weight.ndim != 2 or scale.shape != (weight.shape[0], 1):
        raise ValueError(f"Unexpected embedding shapes: weight {tuple(weight.shape)}, scale {tuple(scale.shape)}")
    output = torch.empty(weight.shape, dtype=torch.bfloat16)
    for start in range(0, weight.shape[0], rows):
        stop = min(start + rows, weight.shape[0])
        output[start:stop] = (weight[start:stop].float() * scale[start:stop]).to(torch.bfloat16)
    return output


def convert_quantized_tensor(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    target = fastvideo_name(name)
    if name.endswith(".weight_scale_2"):
        if tensor.dtype != torch.float32 or tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()) \
                or not bool((tensor > 0).all()):
            raise ValueError(f"{name} must be one finite positive FP32 scalar")
        return target[:-len(".weight_scale_2")] + ".weight_global_scale", tensor.reciprocal().reshape(1)
    if name.endswith(".weight_scale"):
        if tensor.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{name} must contain E4M3 block scales, got {tensor.dtype}")
        return target, tensor.view(torch.uint8).contiguous()
    if name.endswith(".weight"):
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{name} must contain packed uint8 E2M1 values, got {tensor.dtype}")
        # Comfy: HIGH=even, LOW=odd. FlashInfer: LOW=even, HIGH=odd.
        return target[:-len(".weight")] + ".weight_packed", ((tensor << 4) | (tensor >> 4)).contiguous()
    raise ValueError(f"Unsupported quantized tensor {name}")


def main() -> None:
    args = parse_args()
    if not args.src.is_file():
        raise SystemExit(f"Source safetensors file does not exist: {args.src}")

    staging = staging_destination(args.dst)
    writer = ShardWriter(staging, int(args.shard_size_gb * (1 << 30)))
    quantized_count = 0
    copied_count = 0
    try:
        with safe_open(args.src, framework="pt", device="cpu") as handle:
            quantized, pre_scaled = inspect_source(handle)
            embedding = dequantize_embedding(
                handle.get_tensor("model.embed_tokens.weight"),
                handle.get_tensor("model.embed_tokens.weight_scale"),
            )
            writer.add("model.language_model.embed_tokens.weight", embedding)

            input_sizes: dict[str, int] = {}
            for name in sorted(handle.keys()):
                if name.endswith(COMFY_QUANT_SUFFIX) or name in {
                        "model.embed_tokens.weight", "model.embed_tokens.weight_scale"
                }:
                    continue
                prefix = name.rsplit(".", 1)[0]
                tensor = handle.get_tensor(name)
                if prefix in quantized and name.endswith(WEIGHT_SUFFIXES):
                    target_name, target_tensor = convert_quantized_tensor(name, tensor)
                    writer.add(target_name, target_tensor)
                    if name.endswith(".weight"):
                        input_sizes[prefix] = tensor.shape[1] * 2
                        quantized_count += 1
                else:
                    writer.add(fastvideo_name(name), tensor.contiguous())
                    copied_count += 1

            for prefix in sorted(quantized):
                target = fastvideo_name(prefix) + ".pre_quant_scale"
                if prefix in pre_scaled:
                    pre_scale = handle.get_tensor(prefix + ".pre_quant_scale")
                    if pre_scale.dtype != torch.bfloat16 or pre_scale.shape != (input_sizes[prefix], ):
                        raise ValueError(f"Unexpected {prefix}.pre_quant_scale shape or dtype")
                    # It was already copied in the general branch above.
                    continue
                writer.add(target, torch.ones(input_sizes[prefix], dtype=torch.bfloat16))
            writer.finish()

        producer = {
            "converter": Path(__file__).name,
            "source": str(args.src),
            "source_format": "comfy_nvfp4_awq",
            "awq_pre_quant_linears": len(pre_scaled),
        }
        if args.source_revision:
            producer["source_revision"] = args.source_revision
        config = {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "num_hidden_layers_override": EXPECTED_LAYERS,
            "quantization_config": serialized_nvfp4_quantization_config(
                pre_quant_scale=True,
                producer=producer,
            ),
        }
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, args.dst)
    except BaseException:
        writer.abort()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "source": str(args.src),
        "destination": str(args.dst),
        "quantized_linears": quantized_count,
        "awq_pre_quant_linears": len(pre_scaled),
        "identity_pre_quant_linears": quantized_count - len(pre_scaled),
        "copied_tensors": copied_count,
        "output_bytes": writer.total_bytes,
        "shards": writer.shard_index,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
