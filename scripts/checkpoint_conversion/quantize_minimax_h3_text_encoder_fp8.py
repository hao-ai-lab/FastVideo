# SPDX-License-Identifier: Apache-2.0
"""Serialize a MiniMax-H3 (Qwen3-VL) text encoder as block-scaled FP8.

Writes the checkpoint layout read by ``MiniMaxH3SerializedFP8Config``
(``fastvideo/models/encoders/minimax_h3_checkpoint_fp8.py``): every 2-D
language-model linear becomes an E4M3 ``weight`` plus a ``weight_scale_inv``
tensor of shape ``[N / 128, K / 128]``, ``config.json`` gains a
``quantization_config`` block, and the safetensors index is regenerated.

The FP8 encoder is 35.5 GB instead of 66.7 GB. On Blackwell it runs through the
FlashInfer groupwise GEMM; on Hopper (sm_90) each linear dequantizes its weight
to BF16 right before a plain matmul, which is what lets the encoder stay
resident beside an FSDP-sharded DiT on 80 GB GPUs.

Adapted from hlander-ai/minimax-h3 (Apache-2.0).

Usage:
    python scripts/checkpoint_conversion/quantize_minimax_h3_text_encoder_fp8.py \
        --source ./FastH3-Preview-v1/text_encoder \
        --output ./FastH3-TextEncoder-FP8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK_SIZE = 128
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Tensors that keep their checkpoint precision, with the reason each is skipped.
KEEP_PRECISION = {
    "visual": "the vision tower is listed in modules_to_not_convert and stays BF16",
    "embed": "embeddings are gathered, not multiplied, so FP8 buys nothing",
    "lm_head": "never executed by the encoder; kept for checkpoint parity",
}


def _blockwise_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(E4M3 weight, per-128x128-block inverse scales)``."""
    rows, columns = weight.shape
    pad_rows = (BLOCK_SIZE - rows % BLOCK_SIZE) % BLOCK_SIZE
    pad_columns = (BLOCK_SIZE - columns % BLOCK_SIZE) % BLOCK_SIZE
    padded = weight
    if pad_rows or pad_columns:
        padded = torch.nn.functional.pad(weight, (0, pad_columns, 0, pad_rows))
    padded_rows, padded_columns = padded.shape
    blocks = padded.view(padded_rows // BLOCK_SIZE, BLOCK_SIZE, padded_columns // BLOCK_SIZE, BLOCK_SIZE)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_(min=1e-12).float() / E4M3_MAX
    quantized = (blocks.float() / scale[:, None, :, None]).clamp_(-E4M3_MAX, E4M3_MAX)
    quantized = quantized.view(padded_rows, padded_columns)[:rows, :columns]
    return quantized.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def _is_language_linear(key: str, tensor: torch.Tensor) -> bool:
    if not key.endswith(".weight") or tensor.ndim != 2:
        return False
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    return not any(marker in key for marker in KEEP_PRECISION)


def convert(source: Path, output: Path, device: torch.device) -> None:
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no text-encoder safetensors shards under {source}")
    output.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    modules_to_not_convert: set[str] = set()
    quantized_linears = 0
    total_size = 0
    for number, shard in enumerate(shards, start=1):
        converted: dict[str, torch.Tensor] = {}
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open exposes keys(), not iteration
                tensor = handle.get_tensor(key)
                if "visual" in key:
                    modules_to_not_convert.add(key.rsplit(".", 1)[0])
                if _is_language_linear(key, tensor):
                    quantized, scale = _blockwise_fp8(tensor.to(device, torch.bfloat16))
                    converted[key] = quantized.cpu()
                    converted[key.removesuffix("weight") + "weight_scale_inv"] = scale.cpu()
                    quantized_linears += 1
                else:
                    converted[key] = tensor
        destination = output / shard.name
        save_file(converted, str(destination), metadata={"format": "pt"})
        total_size += destination.stat().st_size
        weight_map.update({key: shard.name for key in converted})
        del converted
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"quantized shard {number}/{len(shards)} ({quantized_linears} linears so far)")

    for extra in source.iterdir():
        if extra.is_file() and extra.suffix != ".safetensors" and extra.name != "model.safetensors.index.json":
            shutil.copy2(extra, output / extra.name)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [BLOCK_SIZE, BLOCK_SIZE],
        "modules_to_not_convert": sorted(modules_to_not_convert) or ["visual"],
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} ({total_size / 1e9:.1f} GB, {quantized_linears} FP8 linears)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="BF16 text_encoder directory")
    parser.add_argument("--output", type=Path, required=True, help="destination directory")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device used for the block reductions (default: cuda when available)")
    args = parser.parse_args()
    convert(args.source, args.output, torch.device(args.device))


if __name__ == "__main__":
    main()
