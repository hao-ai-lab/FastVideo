# SPDX-License-Identifier: Apache-2.0
"""Comfy ``int8_tensorwise`` + ConvRot overlay for the MiniMax-H3 video VAE decoder.

The export stores decoder transformer linears as signed int8 with per-output
channel scales and a JSON ``comfy_quant`` marker:

    {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}

Weights were rotated offline by a normalized regular Hadamard (group 256).
Inference rotates activations with the same matrix, row-quantizes them, then
runs int8 GEMM. Encoder convolutions stay dense; only the ViT decoder blocks
are quantized.

Comfy names (``to_qkv``, ``ff.w1`` / ``ff.w2``, ``x_embedder``) are remapped
onto FastVideo's split Q/K/V and ``ff.net`` surface.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.logger import init_logger

logger = init_logger(__name__)

INT8_CONVROT_FILENAME = "minimax_h3_video_vae_int8_convrot.safetensors"
_COM_FY_QUANT_SUFFIX = ".comfy_quant"
_DEFAULT_GROUP_SIZE = 256

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def regular_hadamard(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized regular Hadamard of order ``4**k`` (ConvRot Theorem 3.3)."""
    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    hadamard = h4
    current = 4
    while current < size:
        hadamard = torch.kron(hadamard, h4)
        current *= 4
    hadamard = hadamard / math.sqrt(size)
    _HADAMARD_CACHE[cache_key] = hadamard
    return hadamard


def rotate_activation(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply the block-diagonal ConvRot Hadamard to the last dimension of ``x``."""
    features = x.shape[-1]
    if features % group_size:
        raise ValueError(f"features {features} are not divisible by convrot group_size {group_size}")
    groups = features // group_size
    hadamard = regular_hadamard(group_size, device=x.device, dtype=x.dtype)
    grouped = x.reshape(*x.shape[:-1], groups, group_size)
    return torch.matmul(grouped, hadamard).reshape(x.shape)


def parse_comfy_quant_marker(blob: torch.Tensor) -> dict[str, Any]:
    """Decode the uint8 JSON marker Comfy stores next to each quantized linear."""
    raw = bytes(blob.detach().cpu().contiguous().view(torch.uint8).numpy())
    raw = raw.split(b"\x00", 1)[0]
    marker = json.loads(raw.decode("utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("comfy_quant marker must be a JSON object")
    return marker


class Int8ConvRotLinear(nn.Module):
    """W8A8 linear matching Comfy ``int8_tensorwise`` (+ optional ConvRot)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        convrot: bool,
        group_size: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot = convrot
        self.group_size = group_size
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def _dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.weight.to(dtype) * self.weight_scale.to(dtype)

    @staticmethod
    def _dequant_int8_gemm(
        acc: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        # int32 acc is ~K·127² and overflows fp16 before 1/127 scales land.
        return acc.float() * x_scale.float() * weight_scale.t().float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        if self.convrot:
            x_2d = rotate_activation(x_2d, self.group_size)
        if x_2d.device.type == "cuda" and x_2d.shape[-1] % 8 == 0:
            row_max = x_2d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
            x_scale = row_max / 127.0
            x_q = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)
            # torch._int_mm requires M > 16. VAE decode is far above that;
            # pad only the leftover short rows.
            rows = x_q.shape[0]
            if rows <= 16:
                pad = 17 - rows
                x_q = F.pad(x_q, (0, 0, 0, pad))
                x_scale = F.pad(x_scale, (0, 0, 0, pad))
            acc = torch._int_mm(x_q, self.weight.t().contiguous())[:rows]
            x_scale = x_scale[:rows]
            out = self._dequant_int8_gemm(acc, x_scale, self.weight_scale)
        else:
            out = F.linear(x_2d.float(), self._dequant_weight(torch.float32))
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(dtype=x.dtype).view(*original_shape[:-1], self.out_features)


def _int8_linear_from_tensors(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    marker: dict[str, Any],
) -> Int8ConvRotLinear:
    if weight.dtype != torch.int8:
        raise ValueError(f"expected int8 weight, got {weight.dtype}")
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    convrot = bool(marker.get("convrot", False))
    group_size = int(marker.get("convrot_groupsize", _DEFAULT_GROUP_SIZE)) if convrot else _DEFAULT_GROUP_SIZE
    if marker.get("format") not in (None, "int8_tensorwise"):
        raise ValueError(f"unsupported comfy_quant format {marker.get('format')!r}")
    layer = Int8ConvRotLinear(
        weight.shape[1],
        weight.shape[0],
        bias=bias is not None,
        convrot=convrot and weight.shape[1] % group_size == 0,
        group_size=group_size,
    )
    layer.weight.copy_(weight)
    layer.weight_scale.copy_(scale.to(torch.float32))
    if bias is not None and layer.bias is not None:
        layer.bias.data.copy_(bias)
    return layer


def _split_qkv(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    marker: dict[str, Any],
    *,
    heads: int,
    dim_head: int,
) -> tuple[Int8ConvRotLinear, Int8ConvRotLinear, Int8ConvRotLinear]:
    """Undo Comfy fused QKV: per-head ``[q, k, v]`` then stacked heads."""
    expected = heads * 3 * dim_head
    if weight.shape[0] != expected:
        raise ValueError(f"fused to_qkv out_features {weight.shape[0]} != heads*3*dim_head {expected}")
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    in_features = weight.shape[1]
    weight = weight.view(heads, 3, dim_head, in_features)
    scale = scale.view(heads, 3, dim_head, 1)
    bias_view = None if bias is None else bias.view(heads, 3, dim_head)
    pieces = []
    for index in range(3):
        w = weight[:, index].reshape(heads * dim_head, in_features)
        s = scale[:, index].reshape(heads * dim_head, 1)
        b = None if bias_view is None else bias_view[:, index].reshape(heads * dim_head)
        pieces.append(_int8_linear_from_tensors(w.contiguous(), s.contiguous(), b, marker))
    return pieces[0], pieces[1], pieces[2]


def _swap_swiglu_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Comfy ``[gate; value]`` → FastVideo ``[value; gate]`` along out_features."""
    if tensor.shape[0] % 2:
        raise ValueError(f"SwiGLU packed rows {tensor.shape[0]} are not even")
    half = tensor.shape[0] // 2
    return torch.cat([tensor[half:], tensor[:half]], dim=0)


def overlay_minimax_h3_int8_convrot_decoder(vae: nn.Module, checkpoint_path: str | Path) -> int:
    """Swap H3 VAE decoder transformer linears for Comfy int8-convrot weights.

    Dense decoder tensors that the export still stores in float (embed, norms,
    scales, ``proj_out``) are copied onto the matching FastVideo modules.
    Returns the number of quantized linears installed.
    """
    path = Path(checkpoint_path)
    tensors = safetensors_load_file(str(path))
    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        raise ValueError("MiniMax-H3 VAE overlay expected a `.decoder` module")

    if "decoder.x_embedder.weight" in tensors and hasattr(decoder, "proj_in"):
        decoder.proj_in.weight.data.copy_(tensors["decoder.x_embedder.weight"])
        if decoder.proj_in.bias is not None and "decoder.x_embedder.bias" in tensors:
            decoder.proj_in.bias.data.copy_(tensors["decoder.x_embedder.bias"])
    for name in (
            "decoder.register_tokens",
            "decoder.norm_out.weight",
            "decoder.norm_out.bias",
            "decoder.proj_out.weight",
            "decoder.proj_out.bias",
    ):
        if name not in tensors:
            continue
        module_name, _, param_name = name.removeprefix("decoder.").rpartition(".")
        target = decoder if not module_name else decoder.get_submodule(module_name)
        getattr(target, param_name).data.copy_(tensors[name])

    quantized = 0
    blocks = decoder.transformer_blocks
    for index, block in enumerate(blocks):
        prefix = f"decoder.transformer_blocks.{index}"
        marker = parse_comfy_quant_marker(tensors[f"{prefix}.attn.to_qkv{_COM_FY_QUANT_SUFFIX}"])
        to_q, to_k, to_v = _split_qkv(
            tensors[f"{prefix}.attn.to_qkv.weight"],
            tensors[f"{prefix}.attn.to_qkv.weight_scale"],
            tensors.get(f"{prefix}.attn.to_qkv.bias"),
            marker,
            heads=int(block.attn.heads),
            dim_head=int(block.attn.dim_head),
        )
        block.attn.to_q = to_q
        block.attn.to_k = to_k
        block.attn.to_v = to_v
        quantized += 3
        out_marker = parse_comfy_quant_marker(tensors[f"{prefix}.attn.to_out{_COM_FY_QUANT_SUFFIX}"])
        to_out = _int8_linear_from_tensors(
            tensors[f"{prefix}.attn.to_out.weight"],
            tensors[f"{prefix}.attn.to_out.weight_scale"],
            tensors.get(f"{prefix}.attn.to_out.bias"),
            out_marker,
        )
        block.attn.to_out[0] = to_out
        quantized += 1
        w1_marker = parse_comfy_quant_marker(tensors[f"{prefix}.ff.w1{_COM_FY_QUANT_SUFFIX}"])
        w1_weight = _swap_swiglu_halves(tensors[f"{prefix}.ff.w1.weight"])
        w1_scale = _swap_swiglu_halves(tensors[f"{prefix}.ff.w1.weight_scale"])
        w1_bias = tensors.get(f"{prefix}.ff.w1.bias")
        if w1_bias is not None:
            w1_bias = _swap_swiglu_halves(w1_bias)
        block.ff.net[0].proj = _int8_linear_from_tensors(w1_weight, w1_scale, w1_bias, w1_marker)
        quantized += 1
        w2_marker = parse_comfy_quant_marker(tensors[f"{prefix}.ff.w2{_COM_FY_QUANT_SUFFIX}"])
        block.ff.net[2] = _int8_linear_from_tensors(
            tensors[f"{prefix}.ff.w2.weight"],
            tensors[f"{prefix}.ff.w2.weight_scale"],
            tensors.get(f"{prefix}.ff.w2.bias"),
            w2_marker,
        )
        quantized += 1
        for dense_name in ("norm1.weight", "norm2.weight", "scale1", "scale2"):
            key = f"{prefix}.{dense_name}"
            if key not in tensors:
                continue
            if dense_name.endswith(".weight"):
                module = block.get_submodule(dense_name.rsplit(".", 1)[0])
                module.weight.data.copy_(tensors[key])
            else:
                getattr(block, dense_name).data.copy_(tensors[key])

    device = next(vae.parameters()).device
    vae.decoder.to(device)
    logger.info("Overlaid MiniMax-H3 int8-convrot VAE decoder: %s linears from %s", quantized, path)
    return quantized


def is_int8_convrot_vae_path(path: str | Path) -> bool:
    """Return whether *path* is the Comfy int8-convrot overlay file."""
    raw = os.path.basename(str(path))
    real = os.path.basename(os.path.realpath(str(path)))
    return INT8_CONVROT_FILENAME in (raw, real) or "int8_convrot" in raw or "int8_convrot" in real


def dense_vae_safetensors(paths: list[str]) -> list[str]:
    """Drop the ConvRot overlay so it is not loaded as a dense VAE shard."""
    return [path for path in paths if not is_int8_convrot_vae_path(path)]


def find_int8_convrot_vae_path(model_path: str | Path) -> Path | None:
    """Return the ConvRot export next to a VAE directory, if present."""
    candidate = Path(model_path) / INT8_CONVROT_FILENAME
    return candidate if candidate.is_file() else None
