# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Linum Inc.
# Adapted from https://github.com/Linum-AI/pjit-21k/blob/main/pjit_21k/noise.py
"""Reproduce the release's H100 SXM Philox noise layout on NVIDIA GPUs."""

import math
from functools import lru_cache

import numpy as np
import torch

_BLOCK = 256  # threads per block in torch's RNG launch
_UNROLL = 4  # normals per curand_normal4 call
# curand_globals.h constants, folded in fp32 as nvcc folds them.
_INV_2POW32 = float(np.float32(2.3283064e-10))
_INV_2POW32_2PI = float(np.float32(np.float32(2.3283064e-10) * np.float32(6.2831855)))
_INV_2POW32_HALF = float(np.float32(_INV_2POW32) / np.float32(2))
_INV_2POW32_2PI_HALF = float(np.float32(_INV_2POW32_2PI) / np.float32(2))


@lru_cache(maxsize=1)
def _build_kernel():
    """Import and compile the NVIDIA kernel only when CUDA noise is requested."""
    import triton
    import triton.language as tl

    @triton.jit
    def box_muller(x, y, inv, inv_half, inv_2pi, inv_2pi_half):
        # curand_normal.h _curand_box_muller, device branch.
        u = tl.fma(x.to(tl.float32), inv, inv_half)
        v = tl.fma(y.to(tl.float32), inv_2pi, inv_2pi_half)
        s = tl.sqrt_rn(-2.0 * tl.extra.cuda.libdevice.log(u))
        sin_v = tl.inline_asm_elementwise("sin.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        cos_v = tl.inline_asm_elementwise("cos.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        return sin_v * s, cos_v * s

    @triton.jit
    def randn_kernel(out_ptr, seed, counter0, threads, numel, iterations, inv, inv_half, inv_2pi, inv_2pi_half,
                     BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = idx < threads
        zero = tl.zeros([BLOCK], dtype=tl.uint32)
        subsequence_lo = (idx & 0xFFFFFFFF).to(tl.uint32)
        subsequence_hi = (idx >> 32).to(tl.uint32)
        for it in range(iterations):
            c0 = zero + (counter0 + it).to(tl.uint32)
            r0, r1, r2, r3 = tl.philox(seed, c0, zero, subsequence_lo, subsequence_hi)
            n0, n1 = box_muller(r0, r1, inv, inv_half, inv_2pi, inv_2pi_half)
            n2, n3 = box_muller(r2, r3, inv, inv_half, inv_2pi, inv_2pi_half)
            base = idx + threads * UNROLL * it
            tl.store(out_ptr + base, n0, mask=live & (base < numel))
            tl.store(out_ptr + base + threads, n1, mask=live & (base + threads < numel))
            tl.store(out_ptr + base + 2 * threads, n2, mask=live & (base + 2 * threads < numel))
            tl.store(out_ptr + base + 3 * threads, n3, mask=live & (base + 3 * threads < numel))

    return randn_kernel


def randn_as_on_sm_count(size: tuple[int, ...], generator: torch.Generator, sm_count: int) -> torch.Tensor:
    """Reproduce ATen's CUDA grid-stride normal draw for an H100 with the requested SM count."""
    numel = math.prod(size)
    grid = min(math.ceil(numel / _BLOCK), 2048 // _BLOCK * sm_count)
    threads = _BLOCK * grid
    iterations = (numel - 1) // (threads * _UNROLL) + 1
    offset = generator.get_offset()
    if offset % 4 != 0:
        raise ValueError(f"Generator offset {offset} is not a multiple of 4.")

    out = torch.empty(numel, dtype=torch.float32, device=generator.device)
    with torch.cuda.device(generator.device):
        _build_kernel()[(grid, )](out,
                                  generator.initial_seed(),
                                  offset // 4,
                                  threads,
                                  numel,
                                  iterations,
                                  _INV_2POW32,
                                  _INV_2POW32_HALF,
                                  _INV_2POW32_2PI,
                                  _INV_2POW32_2PI_HALF,
                                  BLOCK=_BLOCK,
                                  UNROLL=_UNROLL)
    generator.set_offset(offset + iterations * 4)  # torch's counter_offset
    return out.view(size)


def pixel_noise(shape: tuple[int, ...], seed: int, device: torch.device, sm_count: int | None) -> torch.Tensor:
    """Draw a single image's initial noise with the reference's 32-bit seed convention."""
    nvidia = device.type == "cuda" and torch.version.hip is None
    # MPS has no CUDA Philox kernel. A CPU stream is reproducible on the Mac.
    rng_device = torch.device("cpu") if device.type == "mps" else device
    generator = torch.Generator(device=rng_device).manual_seed(seed % (1 << 32))
    if nvidia and sm_count is not None:
        return randn_as_on_sm_count(shape, generator, sm_count)
    return torch.randn(shape, generator=generator, dtype=torch.float32, device=rng_device).to(device)
