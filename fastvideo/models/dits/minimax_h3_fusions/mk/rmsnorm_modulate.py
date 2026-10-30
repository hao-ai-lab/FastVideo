# SPDX-License-Identifier: Apache-2.0
"""MotionKernel H3 kernel: Fused H3 RMSNorm + row-indexed AdaLN modulation.

Found with MotionKernel (agent-driven kernel search). Tuned on GB200 / sm_100 for the
MiniMax-H3 inference shapes; opt-in through ``FASTVIDEO_MINIMAX_H3_FUSIONS=mk``.

MiniMax H3 RMSNorm + row-indexed AdaLN modulation (two-kernel fast path).

Math (FastVideo ``minimax_h3.py`` lines 729-731), FP32 internally with one
rounding at the store::

    y = rms_norm(x, weight, eps) * (1 + scale[idx]) + shift[idx]

Shape notes that drive the design (GB200, hidden D = 5376 = 2**8 * 21, so D is
*not* a power of two):

* The vendored Sol-Engine starter runs one program per row with ``BLOCK =
  next_pow2(D) = 8192`` and ``num_warps=16``; 34% of its lanes are masked off
  and the whole 8192-lane tile is live in registers.
* The AdaLN tables are row-strided views (row stride ``6 * D``) with only
  ``table_rows`` (<= 12) distinct rows, so every token re-reads the same
  ``weight``, ``scale`` and ``shift`` vectors: 3 * D * 2 bytes of gather per
  row on top of the D * 2 bytes of activation.

This candidate keeps the public interface and splits the work:

1. ``_fuse_adaln_kernel`` (tiny: ``table_rows`` rows) folds the per-table-row
   modulation into ``a[t] = weight * (1 + scale[t])`` and ``b[t] = shift[t]``,
   stored in the activation dtype. The main pass then gathers two vectors per
   token instead of three, and they are read at the activation width.
2. ``_rmsnorm_modulate_kernel`` is a chunked single-row program with no lane
   padding: ``ceil(D / BLOCK)`` chunks of ``BLOCK`` lanes (BLOCK = 2048 for
   D = 5376, i.e. three chunks with only the last one partially masked). It
   makes a first pass over the row to accumulate the sum of squares, then a
   second pass that re-reads the row (an L1/L2 hit, the row is 10.5 KiB) and
   applies the store. That keeps the live register tile at one chunk instead
   of a whole padded row, which is what the starter loses to.

Performance notes (GB200, measured against a minimal Triton streaming copy of
the same byte count on the same shapes, i.e. the true roofline):

* copy roofline per production shape: 58.9 / 98.6 / 129.1 / 140.8 / 183.9 us
  (6.2-6.9 TB/s), mean 122.3 us.
* this kernel: mean of five shapes ~126.3 us, i.e. ~94.5% of the copy
  roofline re-measured for this attempt (55.6 / 95.1 / 124.7 / 138.0 / 184.6 us,
  mean 119.6 us); on the 59k shape 188.2 us vs 184.6 us (98%).

The step that mattered most over the starter (134.8 us -> 128.5 us) was making
the hidden width a ``tl.constexpr`` and unrolling the chunk loops with
``tl.static_range``: the lane masks, the row offset and the reduction become
static, so every activation/table/store access compiles to an aligned 16-byte
vector access instead of scalar traffic on an unknown-alignment base pointer.
"""

import torch
import triton
import triton.language as tl

H3_NORM_EPS = 1e-5

#: Chunk width of the main pass, and of the table-fusion pass.
_MAIN_BLOCK = 2048
_TABLE_BLOCK = 1024


@triton.jit
def _fuse_adaln_kernel(
    a_ptr,
    b_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    stride_scale_row,
    stride_shift_row,
    NC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (table row, column chunk): a = w * (1 + scale), b = shift."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < NC
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + row * stride_scale_row + cols, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(shift_ptr + row * stride_shift_row + cols, mask=mask, other=0.0).to(tl.float32)
    out = row * NC + cols
    tl.store(a_ptr + out, (w * (1.0 + s)).to(a_ptr.dtype.element_ty), mask=mask)
    tl.store(b_ptr + out, h.to(b_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_modulate_kernel(
    out_ptr,
    x_ptr,
    a_ptr,
    b_ptr,
    index_ptr,
    n_index,
    eps,
    stride_a_row,
    stride_b_row,
    NC: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """One program per row: sumsq pass over the row, then an apply pass.

    ``NC`` (the hidden width) is a ``constexpr`` so that lane masks, row
    offsets and the reduction are fully static: every activation, table and
    store access is then a 16-byte vector access on an aligned base, which is
    what lifts this pass to the streaming-copy roofline. The row is 10.5 KiB,
    so the second pass re-reads it from L1 instead of DRAM.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    table_row = tl.load(index_ptr + row % n_index).to(tl.int64)
    base = row * NC

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for chunk in tl.static_range(NCHUNK):
        off = chunk * BLOCK + cols
        x = tl.load(x_ptr + base + off, mask=off < NC, other=0.0, eviction_policy="evict_last").to(tl.float32)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / NC + eps)

    # Apply the last chunk first: it was just read (still hot in L1) and is the
    # only partially masked one, so its values are kept in registers instead of
    # being re-fetched. Stores use the streaming (``.cs``) cache hint: the
    # output is never re-read, so it should not displace the row/table lines
    # that the apply pass still needs. That combination was worth ~1.5% on the
    # production shapes -- the gap to the streaming-copy roofline is L1/LSU
    # traffic, not DRAM.
    chunk = NCHUNK - 1
    off = chunk * BLOCK + cols
    mask = off < NC
    x = tl.load(x_ptr + base + off, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    a = tl.load(
        a_ptr + table_row * stride_a_row + off,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    b = tl.load(
        b_ptr + table_row * stride_b_row + off,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    y = x * rstd * a + b
    tl.store(
        out_ptr + base + off,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
        cache_modifier=".cs",
    )

    for chunk in tl.static_range(NCHUNK - 1):
        off = chunk * BLOCK + cols
        mask = off < NC
        x = tl.load(x_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + table_row * stride_a_row + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + table_row * stride_b_row + off, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * a + b
        tl.store(
            out_ptr + base + off,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
            cache_modifier=".cs",
        )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _main_block(hidden: int) -> int:
    return min(_MAIN_BLOCK, _next_power_of_two(hidden))


def _num_warps(block: int) -> int:
    if block >= 512:
        return 4
    return 2


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    return table if table.stride(-1) == 1 else table.contiguous()


def rmsnorm_modulate(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """RMSNorm + per-token AdaLN modulation with one activation read and write."""
    if not x.is_cuda:
        raise ValueError("h3_rmsnorm_modulate requires CUDA tensors")
    hidden = x.shape[-1]
    if index.ndim != 1 or index.numel() != x.shape[-2]:
        raise ValueError("index must hold one table row per token")
    if scale.shape[-1] != hidden or shift.shape[-1] != hidden:
        raise ValueError("modulation tables must match the hidden width")
    flat_x = x.reshape(-1, hidden).contiguous()
    weight = weight.contiguous()
    scale = _row_addressable(scale)
    shift = _row_addressable(shift)
    index = index.contiguous()
    output = torch.empty_like(flat_x)

    table_rows = scale.shape[0]
    tab_dtype = flat_x.dtype if flat_x.dtype in (torch.bfloat16, torch.float16) else torch.float32
    fused_a = torch.empty((table_rows, hidden), dtype=tab_dtype, device=flat_x.device)
    fused_b = torch.empty((table_rows, hidden), dtype=tab_dtype, device=flat_x.device)

    table_block = min(_TABLE_BLOCK, _next_power_of_two(hidden))
    _fuse_adaln_kernel[(table_rows, triton.cdiv(hidden, table_block))](
        fused_a,
        fused_b,
        weight,
        scale,
        shift,
        scale.stride(0),
        shift.stride(0),
        NC=hidden,
        BLOCK=table_block,
        num_warps=_num_warps(table_block),
    )

    block = _main_block(hidden)
    _rmsnorm_modulate_kernel[(flat_x.shape[0], )](
        output,
        flat_x,
        fused_a,
        fused_b,
        index,
        index.numel(),
        H3_NORM_EPS,
        fused_a.stride(0),
        fused_b.stride(0),
        NC=hidden,
        BLOCK=block,
        NCHUNK=triton.cdiv(hidden, block),
        num_warps=_num_warps(block),
    )
    return output.view_as(x)
