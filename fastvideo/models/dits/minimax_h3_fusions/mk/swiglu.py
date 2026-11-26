# SPDX-License-Identifier: Apache-2.0
"""MotionKernel H3 kernel: Fused H3 SwiGLU over the packed gate/value halves.

Found with MotionKernel (agent-driven kernel search). Tuned on GB200 / sm_100 for the
MiniMax-H3 inference shapes; opt-in through ``FASTVIDEO_MINIMAX_H3_FUSIONS=mk``.

MiniMax H3 value-first packed SwiGLU -- chunked, unpadded, streaming.

Math (FastVideo ``minimax_h3.py`` @ 7203b4976, lines 188-189)::

    value, gate = hidden_states.chunk(2, dim=-1)   # value first, gate second
    out = value * F.silu(gate)

Kept in FP32 throughout with one rounding at the store (matches Sol-Engine).

Why this beats the vendored Sol-Engine kernel: that kernel runs one program per
row with ``BLOCK = next_pow2(ffn) = 16384`` lanes, so 12.5% of every lane group
is idle and each program keeps two 16384-wide FP32 tiles live (heavy register
pressure, low occupancy). F=14336 = 14 x 1024, so a chunked grid needs no
padding at all. Each program handles one 1024-element column chunk of one row:
one 128-bit vector load per thread per half, FP32 math, one store.

Launch-shape notes measured on GB200 (Pika shapes):
  * A flat 1D grid whose column chunk is ``pid % NTILES`` (so concurrently
    resident CTAs stream adjacent 2 KiB blocks of the same row) is the fastest
    enumeration: ~443 us at 37718 tokens = 7.33 TB/s, versus a measured
    ~7.4 TB/s pure-read / 7.8 TB/s pure-write streaming roofline. Enumerating
    rows fastest costs ~8% (478 us); BLOCK=512 or BLOCK=2048 with more warps
    cost 2-28%.
  * The flat grid also removes the ``grid.y <= 65535`` limit that the 768p
    243-frame shape (73394 rows) exceeds: adding a row-group loop there cost
    4% (888 us vs 852 us).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_I32_LIMIT = 0x7FFFFFFF


@triton.jit
def _h3_swiglu_kernel(
    out_ptr,
    x_ptr,
    ffn_dim,
    n_rows,
    stride_in_row,
    stride_out_row,
    NTILES: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
    GUARD: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    col = pid % NTILES
    row = pid // NTILES
    cols = col * BLOCK + tl.arange(0, BLOCK)
    if I64:
        row = row.to(tl.int64)
    if GUARD:
        if row < n_rows:
            _body(out_ptr, x_ptr, ffn_dim, row, cols, stride_in_row, stride_out_row, BLOCK, EXACT)
    else:
        _body(out_ptr, x_ptr, ffn_dim, row, cols, stride_in_row, stride_out_row, BLOCK, EXACT)


@triton.jit
def _body(out_ptr, x_ptr, ffn_dim, row, cols, stride_in_row, stride_out_row, BLOCK: tl.constexpr, EXACT: tl.constexpr):
    in_off = row * stride_in_row + cols
    if EXACT:
        value = tl.load(x_ptr + in_off).to(tl.float32)
        gate = tl.load(x_ptr + in_off + ffn_dim).to(tl.float32)
        out = value * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty))
    else:
        mask = cols < ffn_dim
        value = tl.load(x_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + in_off + ffn_dim, mask=mask, other=0.0).to(tl.float32)
        out = value * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """Launch the chunked unpadded SwiGLU."""
    if not x.is_cuda:
        raise ValueError("h3_swiglu requires a CUDA tensor")
    packed_width = x.shape[-1]
    if packed_width == 0 or packed_width % 2:
        raise ValueError("h3_swiglu needs a positive even last dimension")
    ffn_dim = packed_width // 2
    flat = x.reshape(-1, packed_width)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    output_shape = (*x.shape[:-1], ffn_dim)
    rows = flat.shape[0]
    if rows == 0:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    out = torch.empty((rows, ffn_dim), dtype=x.dtype, device=x.device)

    block = min(_BLOCK, triton.next_power_of_2(ffn_dim))
    exact = (ffn_dim % block) == 0
    ntiles = (ffn_dim + block - 1) // block
    i64 = rows * packed_width > _I32_LIMIT
    num_warps = _NUM_WARPS if block >= 1024 else 2
    _h3_swiglu_kernel[(ntiles * rows, )](
        out,
        flat,
        ffn_dim,
        rows,
        flat.stride(0),
        out.stride(0),
        NTILES=ntiles,
        BLOCK=block,
        EXACT=exact,
        GUARD=False,
        I64=i64,
        num_warps=num_warps,
    )
    return out.view(output_shape)
