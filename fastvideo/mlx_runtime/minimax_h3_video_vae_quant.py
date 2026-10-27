# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""Weight-only quantization for the MLX MiniMax-H3 ViT video decoder.

The 216 transformer-block linears of the decoder (``attn.to_{q,k,v}``,
``attn.to_out.0``, ``ff.net.0.proj``, ``ff.net.2``) dominate its bytes. They can
be stored in any ``mx.quantize`` mode and are dispatched through
``mx.quantized_matmul``; everything else (norms, ``proj_in``/``proj_out``,
register tokens, ``post_quant_conv``, biases) stays dense.

NVFP4 needs a per-tensor FP32 scale, which ``mx.quantized_matmul`` has no
argument for: like the NVFP4 text encoder, the weight is scaled by
``gs = 448 * 6 / amax(w)`` before block quantization (so the E4M3 block scales
use their full range) and the matmul output is divided by ``gs``. That is
exactly the FlashInfer convention of the FastH3 NVFP4 decoder release, so the
pre-packed ``decoder-nvfp4.safetensors`` loads without requantizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import mlx.core as mx

QUANT_MODES: dict[str, tuple[str, int, int]] = {
    "nvfp4": ("nvfp4", 16, 4),
    "mxfp8": ("mxfp8", 32, 8),
    "int8": ("affine", 64, 8),
    "int4": ("affine", 64, 4),
}

NVFP4_GLOBAL_SCALE_NUMERATOR = 448.0 * 6.0  # E4M3 max * E2M1 max
BLOCK_LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


@dataclass(frozen=True)
class QuantizedWeight:
    """A packed linear weight; ``output_scale`` undoes the pre-quantization tensor scale."""

    weight: Any
    scales: Any
    biases: Any
    mode: str
    group_size: int
    bits: int
    output_scale: float = 1.0

    def matmul(self, x):
        y = mx.quantized_matmul(x,
                                self.weight,
                                self.scales,
                                self.biases,
                                transpose=True,
                                group_size=self.group_size,
                                bits=self.bits,
                                mode=self.mode)
        return y if self.output_scale == 1.0 else y * self.output_scale

    def dequantize(self, dtype=mx.float32):
        w = mx.dequantize(self.weight,
                          self.scales,
                          self.biases,
                          group_size=self.group_size,
                          bits=self.bits,
                          mode=self.mode,
                          dtype=dtype)
        return w * self.output_scale if self.output_scale != 1.0 else w

    @property
    def nbytes(self) -> int:
        return self.weight.nbytes + self.scales.nbytes + (self.biases.nbytes if self.biases is not None else 0)


def quantize_weight(weight, mode: str) -> QuantizedWeight:
    """Quantize a dense (out, in) matrix; ``mode`` is a key of ``QUANT_MODES``."""
    if mode not in QUANT_MODES:
        raise ValueError(f"Unknown decoder quantization {mode!r}; choose from {sorted(QUANT_MODES)}.")
    mlx_mode, group_size, bits = QUANT_MODES[mode]
    if weight.shape[-1] % group_size:
        raise ValueError(f"Input width {weight.shape[-1]} is not divisible by group size {group_size}.")
    dense = weight.astype(mx.float32)
    output_scale = 1.0
    if mode == "nvfp4":
        amax = float(mx.max(mx.abs(dense)))
        if amax <= 0.0:
            raise ValueError("Cannot NVFP4-quantize an all-zero weight.")
        global_scale = NVFP4_GLOBAL_SCALE_NUMERATOR / amax
        dense = dense * global_scale
        output_scale = 1.0 / global_scale
    packed = mx.quantize(dense if mode == "nvfp4" else weight, group_size=group_size, bits=bits, mode=mlx_mode)
    biases = packed[2] if len(packed) == 3 else None
    return QuantizedWeight(packed[0], packed[1], biases, mlx_mode, group_size, bits, output_scale)


def unswizzle_nvfp4_scales(scale: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """FlashInfer 128x4 scale bytes -> MLX row-major group-16 scale bytes."""
    pad_rows, pad_cols = -(-rows // 128) * 128, -(-cols // 4) * 4
    if scale.size < pad_rows * pad_cols:
        raise ValueError(f"NVFP4 scales need {pad_rows * pad_cols} bytes, got {scale.size}.")
    tiles = scale.reshape(-1)[:pad_rows * pad_cols].reshape(pad_rows // 128, pad_cols // 4, 32, 4, 4)
    return np.ascontiguousarray(tiles.transpose(0, 3, 2, 1, 4).reshape(pad_rows, pad_cols)[:rows, :cols])


def packed_nvfp4_weight(packed_weight, inv_scale, global_sf) -> QuantizedWeight:
    """One linear of a ``fastvideo_h3_decoder_nvfp4_packed_v1`` file as an MLX NVFP4 weight."""
    packed = np.ascontiguousarray(np.asarray(packed_weight.view(mx.uint8)))
    rows, cols = packed.shape[0], packed.shape[1] * 2
    scales = unswizzle_nvfp4_scales(np.asarray(inv_scale.view(mx.uint8)).reshape(-1), rows, cols // 16)
    global_scale = float(global_sf.reshape(-1)[0])
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        raise ValueError(f"Invalid NVFP4 global scale: {global_scale}")
    return QuantizedWeight(mx.array(packed).view(mx.uint32), mx.array(scales), None, "nvfp4", 16, 4, 1.0 / global_scale)


def is_block_linear_weight(key: str) -> bool:
    """True for the weight of one of the 216 decoder transformer-block linears."""
    prefix = "decoder.transformer_blocks."
    if not key.startswith(prefix) or not key.endswith(".weight"):
        return False
    return key[len(prefix):-len(".weight")].split(".", 1)[-1] in BLOCK_LINEARS
