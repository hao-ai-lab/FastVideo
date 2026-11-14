# SPDX-License-Identifier: Apache-2.0
"""Fuse compact key normalization with the GB10 ATen mean reduction tree.

For 18,432 rows and 56 head-128 outputs, ATen uses four accumulator lanes,
four row-reduction warps and 18 CTAs per output. Preserve those additions and
the BF16 normalization boundary while eliminating the separate full-key read.
The caller restricts this specialization to the measured one-Spark geometry.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_normalize_partials(
    X, W, C, S, Index, Y, Partial, ROWS: tl.constexpr, HEADS: tl.constexpr,
    INPUT_ROWS: tl.constexpr, ROTARY: tl.constexpr, EPS: tl.constexpr, CTAS: tl.constexpr,
):
    cta, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    lanes = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 128)[None, :]
    accumulators = tl.full((16, 128), 0, tl.float32)
    for iteration in range(tl.cdiv(ROWS, 16 * CTAS)):
        # ATen input_idx = warp + cta * 4, step_input = 4 * CTAS;
        # its four independent accumulators advance by step_input * 4.
        row = lanes % 4 + cta * 4 + (lanes // 4) * (4 * CTAS) + iteration * (16 * CTAS)
        source = tl.load(Index + row, row < ROWS, other=-1)
        offset = ((batch * INPUT_ROWS + source) * HEADS + head) * 128 + cols
        x = tl.load(X + offset, (row < ROWS) & (source >= 0), other=0).to(tl.float32)
        squared = x * x
        # Original four-warp normalization order, independent of this launch's layout.
        for shift in tl.static_range(5):
            other = tl.gather(squared, tl.broadcast_to(cols ^ (16 >> shift), (16, 128)), axis=1)
            squared = squared + other
        for shift in tl.static_range(2):
            other = tl.gather(squared, tl.broadcast_to(cols ^ (64 >> shift), (16, 128)), axis=1)
            squared = squared + other
        variance = tl.gather(squared, tl.full((16, 1), 0, tl.int32), axis=1) / 128
        normalized = x * tl.rsqrt(variance + EPS) * tl.load(W + cols).to(tl.float32)
        partner = tl.where(cols < ROTARY // 2, cols + ROTARY // 2, cols - ROTARY // 2)
        partner_normalized = tl.gather(normalized, tl.broadcast_to(partner, (16, 128)), axis=1)
        rotated = tl.where(cols < ROTARY // 2, -partner_normalized, partner_normalized)
        rotary_mask = (row < ROWS) & (cols < ROTARY)
        cos = tl.load(C + row * ROTARY + cols, rotary_mask, other=1).to(tl.float32)
        sin = tl.load(S + row * ROTARY + cols, rotary_mask, other=0).to(tl.float32)
        output = tl.where(cols < ROTARY, normalized * cos + rotated * sin, normalized).to(tl.bfloat16)
        tl.store(Y + ((batch * ROWS + row) * HEADS + head) * 128 + cols, output, row < ROWS)
        # Mean consumes the stored BF16 values, not the normalization accumulator.
        accumulators = accumulators + tl.where(row < ROWS, output.to(tl.float32), 0.0)

    warp = tl.arange(0, 4)[:, None]
    sums = tl.gather(accumulators, tl.broadcast_to(warp, (4, 128)), axis=0)
    for accumulator in tl.static_range(1, 4):
        sums = sums + tl.gather(accumulators, tl.broadcast_to(warp + accumulator * 4, (4, 128)), axis=0)
    s0 = tl.gather(sums, tl.full((1, 128), 0, tl.int32), axis=0)
    s1 = tl.gather(sums, tl.full((1, 128), 1, tl.int32), axis=0)
    s2 = tl.gather(sums, tl.full((1, 128), 2, tl.int32), axis=0)
    s3 = tl.gather(sums, tl.full((1, 128), 3, tl.int32), axis=0)
    tl.store(Partial + ((batch * CTAS + cta) * HEADS + head) * 128 + cols, (s0 + s2) + (s1 + s3))


@triton.jit
def _finish_mean(Partial, Mean, HEADS: tl.constexpr, ROWS: tl.constexpr, CTAS: tl.constexpr):
    head, batch = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, 128)
    s0 = tl.full((128,), 0, tl.float32)
    s1, s2, s3 = s0, s0, s0
    for i in tl.static_range(tl.cdiv(CTAS, 4)):
        s0 = s0 + tl.load(Partial + ((batch * CTAS + i * 4) * HEADS + head) * 128 + cols,
                          i * 4 < CTAS, other=0)
        s1 = s1 + tl.load(Partial + ((batch * CTAS + i * 4 + 1) * HEADS + head) * 128 + cols,
                          i * 4 + 1 < CTAS, other=0)
        s2 = s2 + tl.load(Partial + ((batch * CTAS + i * 4 + 2) * HEADS + head) * 128 + cols,
                          i * 4 + 2 < CTAS, other=0)
        s3 = s3 + tl.load(Partial + ((batch * CTAS + i * 4 + 3) * HEADS + head) * 128 + cols,
                          i * 4 + 3 < CTAS, other=0)
    tl.store(Mean + (batch * HEADS + head) * 128 + cols, ((s0 + s2) + (s1 + s3)) * (1.0 / ROWS))


def gather_key_norm_mean(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    eps: float, input_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return original BF16 normalized keys and their ATen-equivalent BF16 mean."""
    if (x.ndim != 4 or x.shape[0] != 1 or x.shape[2:] != (56, 128)
            or not x.is_cuda or x.dtype != torch.bfloat16 or input_index.numel() != 18432
            or weight.shape != (128,) or cos.shape != (18432, 96) or sin.shape != cos.shape
            or input_index.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Fused key mean requires CUDA BF16 batch-1/head-56, 18432 rows and rotary width 96")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("Fused key mean requires eager inference")
    x, weight, cos, sin, input_index = (t.contiguous() for t in (x, weight, cos, sin, input_index))
    rows, heads, dim, ctas = 18432, 56, 128, 18
    out = torch.empty((1, rows, heads, dim), dtype=x.dtype, device=x.device)
    partial = torch.empty((1, ctas, heads, dim), dtype=torch.float32, device=x.device)
    mean = torch.empty((1, 1, heads, dim), dtype=x.dtype, device=x.device)
    _gather_normalize_partials[(ctas, heads, 1)](
        x, weight, cos, sin, input_index, out, partial, rows, heads, x.shape[1], 96, eps, ctas, num_warps=4,
    )
    _finish_mean[(heads, 1)](partial, mean, heads, rows, ctas)
    return out, mean
