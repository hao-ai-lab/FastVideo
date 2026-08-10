# SPDX-License-Identifier: Apache-2.0
"""Torch fake-quant twin for MLX ``mode="mxfp4"`` quantization.

MLX's MXFP4 format uses 32-value groups, one E8M0 power-of-two scale per
group, and signed E2M1 FP4 element codes.  This module provides a torch-side
reference for QAT experiments so training can model the same quantization grid
used by the Apple/MLX deployment runtime.
"""

from __future__ import annotations

import torch

DEFAULT_GROUP_SIZE = 32

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _check_group_size(weight: torch.Tensor, group_size: int) -> None:
    if group_size != DEFAULT_GROUP_SIZE:
        raise ValueError(f"MLX MXFP4 uses fixed group_size {DEFAULT_GROUP_SIZE}, got {group_size}")
    if weight.shape[-1] % group_size != 0:
        raise ValueError(f"Last dimension {weight.shape[-1]} is not divisible by {group_size}")


def _scale_exponent(max_abs: torch.Tensor) -> torch.Tensor:
    # MLX stores an E8M0 scale byte. Empirically this is round-to-nearest-even
    # log2(max_abs / 6), then biased by 127.  Clamp zero groups to the smallest
    # normal-ish exponent used here to avoid log2(0); all codes will be zero.
    safe = torch.clamp(max_abs / 6.0, min=2.0**-127)
    return torch.round(torch.log2(safe)).to(torch.int32)


def _quantize_e2m1_codes(scaled_abs: torch.Tensor) -> torch.Tensor:
    table = _E2M1_VALUES.to(device=scaled_abs.device)
    distances = (scaled_abs[..., None].float() - table).abs()
    # Ties choose the even FP4 code: 0.25 -> 0, 0.75 -> 1.0, etc.  The tiny
    # odd-code penalty is far below the spacing between non-tie candidates.
    code_penalty = (torch.arange(8, device=scaled_abs.device) % 2).float() * 1e-7
    return torch.argmin(distances + code_penalty, dim=-1).to(torch.int32)


def mlx_mxfp4_quantize_reference(
    weight: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(codes, scale_bytes)`` matching MLX MXFP4 quantization."""
    _check_group_size(weight, group_size)
    original_shape = weight.shape
    groups = weight.float().reshape(*original_shape[:-1], -1, group_size)
    max_abs = groups.abs().amax(dim=-1)
    exponent = _scale_exponent(max_abs)
    scale = torch.pow(2.0, exponent.float()).to(groups.device)
    scaled_abs = groups.abs() / scale[..., None]
    magnitude_codes = _quantize_e2m1_codes(scaled_abs)
    sign = (groups < 0).to(torch.int32) * 8
    codes = (magnitude_codes + sign).reshape(original_shape)
    scale_bytes = (exponent + 127).to(torch.uint8)
    return codes, scale_bytes


def mlx_mxfp4_dequantize_reference(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_shape: torch.Size | tuple[int, ...],
    group_size: int = DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize E2M1 ``codes`` and E8M0 ``scales`` to float32 values."""
    if out_shape[-1] % group_size != 0:
        raise ValueError(f"Last dimension {out_shape[-1]} is not divisible by {group_size}")
    table = _E2M1_VALUES.to(device=codes.device)
    grouped_codes = codes.reshape(*out_shape[:-1], -1, group_size).to(torch.int64)
    sign = torch.where(grouped_codes >= 8, -1.0, 1.0).to(codes.device)
    magnitude = table[(grouped_codes & 0x7).long()]
    exponent = scales.to(torch.int32) - 127
    scale = torch.pow(2.0, exponent.float()).to(codes.device)
    dequant = sign * magnitude * scale[..., None]
    return dequant.reshape(out_shape).float()


def fake_quantize_mlx_mxfp4(
    weight: torch.Tensor,
    *,
    simulate_dtype: torch.dtype = torch.float16,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Fake-quantize ``weight`` with STE gradients for MXFP4 QAT."""
    simulated = weight.detach().to(simulate_dtype)
    codes, scales = mlx_mxfp4_quantize_reference(
        simulated,
        group_size=group_size,
    )
    dequant = mlx_mxfp4_dequantize_reference(
        codes,
        scales,
        out_shape=weight.shape,
        group_size=group_size,
    ).to(simulate_dtype)
    simulated_weight = weight.to(simulate_dtype)
    return simulated_weight + (dequant - simulated_weight).detach()
