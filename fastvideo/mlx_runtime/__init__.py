# SPDX-License-Identifier: Apache-2.0
"""Experimental Apple MLX runtime helpers.

This package is intentionally small for now. It exists to grow the Apple-native
FastWan path in measurable steps: shape planning, primitive benchmarks, then
Wan block parity, then full DiT/runtime support.
"""

from fastvideo.mlx_runtime.fastwan import (
    FastWanShape,
    MLXQuantizationSpec,
    MLXWanDiT,
    MLXWanTransformerBlock,
    UnsupportedMLXQuantizationError,
    ensure_quantization_supported,
    fastwan_shape,
    fastwan_shape_from_config,
    mlx_dit_from_diffusers_safetensors,
    mlx_block_weights_from_torch,
    mlx_block_weights_from_diffusers_safetensors,
    quantization_support_error,
)
from fastvideo.mlx_runtime.checkpoint import (
    load_mlx_dit_checkpoint,
    save_mlx_dit_checkpoint,
)
from fastvideo.mlx_runtime.memory import (
    AppliedMemoryLimits,
    add_memory_limit_args,
    apply_memory_limits,
    gib_to_bytes,
)

__all__ = [
    "AppliedMemoryLimits",
    "FastWanShape",
    "MLXQuantizationSpec",
    "MLXWanDiT",
    "MLXWanTransformerBlock",
    "UnsupportedMLXQuantizationError",
    "add_memory_limit_args",
    "apply_memory_limits",
    "ensure_quantization_supported",
    "fastwan_shape",
    "fastwan_shape_from_config",
    "gib_to_bytes",
    "load_mlx_dit_checkpoint",
    "mlx_dit_from_diffusers_safetensors",
    "mlx_block_weights_from_diffusers_safetensors",
    "mlx_block_weights_from_torch",
    "quantization_support_error",
    "save_mlx_dit_checkpoint",
]
