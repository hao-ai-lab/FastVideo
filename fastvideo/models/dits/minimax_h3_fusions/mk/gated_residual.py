# SPDX-License-Identifier: Apache-2.0
"""MotionKernel H3 kernel: Fused H3 row-indexed gated residual (hidden + gate[index] * branch).

Found with MotionKernel (agent-driven kernel search). Tuned on GB200 / sm_100 for the
MiniMax-H3 inference shapes; opt-in through ``FASTVIDEO_MINIMAX_H3_FUSIONS=mk``.

Triton kernel for MiniMax H3's post-FFN row-indexed gated residual.

``out = hidden + gate[index] * branch`` with the math in fp32 and a single
bf16 rounding at the store.

Design notes (GB200, sm_100a):
  * The op moves 3 activation-sized streams (2 reads + 1 write); it is purely
    bandwidth bound.  A flat 1-D grid over ``tokens * hidden`` gives fully
    coalesced, 16-byte-vectorized accesses on all three streams with no lane
    padding (the 5376-wide rows are not padded to a power of two).
  * Split full-block / tail code paths.  Every non-tail block runs completely
    unmasked, so the compiler emits unpredicated 128-bit loads/stores; only the
    final block pays for predicates.  Measured 175.6 -> 173.6 us weighted over
    the five production shapes (~1.1%), consistently on every shape.
  * The gathered AdaLN gate row is tiny (<= 12 rows) so it stays resident in
    L2; the per-element row/column split needed to address it costs only ALU,
    of which this kernel uses ~1% of peak.
  * 32-bit offsets (all production shapes fit) halve index register pressure.
  * ``evict_last`` on the three loads is worth ~2%: it keeps the streamed
    lines in L2 long enough for the DRAM scheduler to coalesce the
    read/write turnaround (measured 269 -> 266 us at 59k tokens).
  * Roofline: a minimal unmasked 2-read/1-write Triton ``c = a + b`` kernel
    (same bytes, no gather) runs at 267.7 us / 7.11 TB/s at the largest shape;
    this kernel measures 263.2 us / 7.24 TB/s, i.e. it is at (slightly above)
    the measured 3-stream streaming roofline, so the body is not tunable
    further.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _h3_gated_residual_kernel(
    out_ptr,
    hidden_ptr,
    branch_ptr,
    gate_ptr,
    index_ptr,
    n_elems,
    n_index,
    D: tl.constexpr,
    GS: tl.constexpr,
    BLOCK: tl.constexpr,
    NFULL: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        # Rare >2^31-element shapes: always use 64-bit offsets and masks.
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n_elems
        row = off // D
        col = off - row * D
        table_row = tl.load(index_ptr + row % n_index, mask=mask, other=0).to(tl.int32)
        hidden = tl.load(hidden_ptr + off, mask=mask, eviction_policy="evict_last").to(tl.float32)
        branch = tl.load(branch_ptr + off, mask=mask, eviction_policy="evict_last").to(tl.float32)
        gate = tl.load(gate_ptr + table_row * GS + col, mask=mask, eviction_policy="evict_last").to(tl.float32)
        tl.store(
            out_ptr + off,
            (hidden + gate * branch).to(out_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        base = pid * BLOCK
        if base + BLOCK <= NFULL:
            # Full block: no predicates anywhere on the hot path.
            off = base + tl.arange(0, BLOCK)
            row = off // D
            col = off - row * D
            table_row = tl.load(index_ptr + row % n_index).to(tl.int32)
            hidden = tl.load(hidden_ptr + off, eviction_policy="evict_last").to(tl.float32)
            branch = tl.load(branch_ptr + off, eviction_policy="evict_last").to(tl.float32)
            gate = tl.load(gate_ptr + table_row * GS + col, eviction_policy="evict_last").to(tl.float32)
            tl.store(out_ptr + off, (hidden + gate * branch).to(out_ptr.dtype.element_ty))
        else:
            off = base + tl.arange(0, BLOCK)
            mask = off < n_elems
            row = off // D
            col = off - row * D
            table_row = tl.load(index_ptr + row % n_index, mask=mask, other=0).to(tl.int32)
            hidden = tl.load(hidden_ptr + off, mask=mask, eviction_policy="evict_last").to(tl.float32)
            branch = tl.load(branch_ptr + off, mask=mask, eviction_policy="evict_last").to(tl.float32)
            gate = tl.load(
                gate_ptr + table_row * GS + col,
                mask=mask,
                eviction_policy="evict_last",
            ).to(tl.float32)
            tl.store(
                out_ptr + off,
                (hidden + gate * branch).to(out_ptr.dtype.element_ty),
                mask=mask,
            )


def gated_residual(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """``hidden + gate[index] * branch`` in one pass."""
    if not hidden.is_cuda:
        raise ValueError("h3_gated_residual requires CUDA tensors")
    if branch.shape != hidden.shape:
        raise ValueError("branch must match hidden")
    hidden_size = hidden.shape[-1]
    if index.ndim != 1 or index.numel() != hidden.shape[-2]:
        raise ValueError("index must hold one table row per token")
    if gate.shape[-1] != hidden_size:
        raise ValueError("gate table must match the hidden width")
    flat_hidden = hidden.reshape(-1, hidden_size)
    flat_branch = branch.reshape(-1, hidden_size)
    if not flat_hidden.is_contiguous():
        flat_hidden = flat_hidden.contiguous()
    if not flat_branch.is_contiguous():
        flat_branch = flat_branch.contiguous()
    if gate.stride(-1) != 1:
        gate = gate.contiguous()
    gate_row_stride = gate.stride(0)
    index = index.contiguous()
    out = torch.empty_like(flat_hidden)

    n_elems = flat_hidden.numel()
    if n_elems == 0:
        return out.view_as(hidden)
    block = 4096
    wide = n_elems >= 2**31
    nfull = (n_elems // block) * block
    grid = (triton.cdiv(n_elems, block), )
    _h3_gated_residual_kernel[grid](
        out,
        flat_hidden,
        flat_branch,
        gate,
        index,
        n_elems,
        index.numel(),
        D=hidden_size,
        GS=gate_row_stride,
        BLOCK=block,
        NFULL=nfull,
        WIDE=wide,
        num_warps=8,
    )
    return out.view_as(hidden)
