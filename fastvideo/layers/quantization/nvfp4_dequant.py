# SPDX-License-Identifier: Apache-2.0
"""One-pass serialized NVFP4 weight expansion for BF16 consumer-GPU compute."""
import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_nvfp4(P, S, OUT, N: tl.constexpr, K: tl.constexpr, INVERSE_SCALE, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N * K
    row, col = offsets // K, offsets % K
    packed = tl.load(P + row * (K // 2) + col // 2, valid, other=0).to(tl.uint32)
    code = (packed >> ((col % 2) * 4)) & 15
    magnitude = (code & 7).to(tl.float32)
    value = tl.where(magnitude < 4, magnitude * 0.5, tl.where(magnitude < 6, magnitude - 2, magnitude * 2 - 8))
    value = value * tl.where((code & 8) != 0, -1.0, 1.0)
    group = col // 16
    # FlashInfer layout_128x4: [row_tile, col_tile, row%32, row//32%4, col%4].
    scale_index = ((((row // 128) * (K // 64) + group // 4) * 32 + row % 32) * 4 + (row // 32) % 4) * 4 + group % 4
    scale = tl.load(S + scale_index, valid, other=0.0).to(tl.float32)
    output = (value * scale) * INVERSE_SCALE
    tl.store(OUT + offsets, output, valid)


def dequantize_nvfp4_cuda(packed: torch.Tensor,
                          scales: torch.Tensor,
                          global_scale: float,
                          dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Expand E2M1 nibbles and swizzled E4M3 scales without full FP32 intermediates."""
    if not packed.is_cuda or scales.device != packed.device:
        raise ValueError("NVFP4 fused dequantization requires tensors on the same CUDA device")
    if packed.ndim != 2 or packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("NVFP4 fused dequantization requires packed uint8 weights and scales")
    if not packed.is_contiguous() or not scales.is_contiguous():
        raise ValueError("NVFP4 fused dequantization requires contiguous tensors")
    rows, cols = packed.shape[0], packed.shape[1] * 2
    if rows % 128 or cols % 64 or scales.numel() != rows * cols // 16:
        raise ValueError("NVFP4 fused dequantization requires exact 128x4 scale geometry")
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("NVFP4 fused dequantization requires a floating output dtype")
    output = torch.empty((rows, cols), dtype=dtype, device=packed.device)
    _dequantize_nvfp4[(triton.cdiv(rows * cols, 1024), )](
        packed,
        scales.view(torch.float8_e4m3fn),
        output,
        rows,
        cols,
        # Match Torch's CPU-scalar division: form the
        # reciprocal in double, then cast to FP32.
        1.0 / global_scale,
        BLOCK=1024,
        num_warps=4)
    return output
