# SPDX-License-Identifier: Apache-2.0
"""Fused per-head RMSNorm and partial rotary embedding for MiniMax H3."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only in environments without Triton
    triton = None
    tl = None
    HAVE_TRITON = False


if HAVE_TRITON:
    from .swiglu_quant import _rcp

    @triton.jit
    def _qknorm_partial_rope_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        head_dim,
        rotary_dim,
        half_rotary_dim,
        num_heads,
        seq_len,
        eps,
        BLOCK_SIZE: tl.constexpr,
        ROWS: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        codes_ptr=None,
        scales_ptr=None,
        PACK_FP4: tl.constexpr = False,
        START: tl.constexpr = 0,
        input_index_ptr=None,
        INPUT_ROWS: tl.constexpr = 0,
        GATHER_INPUT: tl.constexpr = False,
    ):
        # int64, like the sibling kernels: with int32 program ids,
        # ``row * head_dim`` wraps once the flattened input reaches 2**31
        # elements (H3's 56 heads x 128 head_dim crosses that at
        # batch*seq >= 299_593 tokens per rank) and the loads/stores below
        # become out-of-bounds. ``seq_index`` inherits int64 from ``row``.
        # Keep each row's four-warp reduction order while amortizing CTA scheduling.
        for i in tl.static_range(ROWS_PER_PROGRAM):
            row = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM + i
            seq_index = (row // num_heads) % seq_len
            cols = tl.arange(0, BLOCK_SIZE)
            head_mask = (cols < head_dim) & (row < ROWS)
            row_offset = row * head_dim
            if GATHER_INPUT:
                source = tl.load(input_index_ptr + seq_index, row < ROWS, other=-1).to(tl.int64)
                input_offset = (((row // (seq_len * num_heads)) * INPUT_ROWS + source) * num_heads
                                + row % num_heads) * head_dim
                input_valid = source >= 0
            else:
                input_offset, input_valid = row_offset, True
            x = tl.load(x_ptr + input_offset + cols, mask=head_mask & input_valid, other=0.0).to(tl.float32)
            variance = tl.sum(x * x, axis=0) / head_dim
            inv_rms = tl.math.rsqrt(variance + eps)
            weight = tl.load(weight_ptr + cols, mask=head_mask, other=0.0).to(tl.float32)
            normalized = x * inv_rms * weight

            rotary_mask = (cols < rotary_dim) & (row < ROWS)
            first_half = cols < half_rotary_dim
            partner_col = tl.where(first_half, cols + half_rotary_dim, cols - half_rotary_dim)
            partner_x = tl.load(
                x_ptr + input_offset + partner_col,
                mask=rotary_mask & input_valid,
                other=0.0,
            ).to(tl.float32)
            partner_weight = tl.load(weight_ptr + partner_col, mask=rotary_mask, other=0.0).to(tl.float32)
            partner_normalized = partner_x * inv_rms * partner_weight
            rotated = tl.where(first_half, -partner_normalized, partner_normalized)

            table_offset = seq_index * rotary_dim + cols
            cos = tl.load(cos_ptr + table_offset, mask=rotary_mask, other=1.0).to(tl.float32)
            sin = tl.load(sin_ptr + table_offset, mask=rotary_mask, other=0.0).to(tl.float32)
            rotary_output = normalized * cos + rotated * sin
            output = tl.where(rotary_mask, rotary_output, normalized)
            tl.store(out_ptr + row_offset + cols, output.to(out_ptr.dtype.element_ty), mask=head_mask)
            if PACK_FP4:
                rounded = output.to(tl.bfloat16).to(tl.float32)
                grouped = tl.reshape(rounded, (BLOCK_SIZE // 16, 16))
                maximum = tl.max(tl.abs(grouped), 1)
                sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
                inv = tl.where(sf.to(tl.float32) == 0, 0, _rcp(sf.to(tl.float32)))
                values = tl.reshape(grouped * inv[:, None], (BLOCK_SIZE // 2, 2))
                lo, hi = tl.split(values)
                packed = tl.inline_asm_elementwise('''{
                    .reg .b8 q;
                    cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
                    mov.b32 $0, {q,q,q,q};
                }''', constraints='=r,f,f', args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
                batch = row // (seq_len * num_heads)
                head = row % num_heads
                c4 = tl.arange(0, BLOCK_SIZE // 2)
                offset = ((batch * num_heads + head) * (seq_len - START) + seq_index - START) * (head_dim // 2)
                tl.store(codes_ptr + offset + c4, packed, (row < ROWS) & (seq_index >= START) & (c4 < head_dim // 2))
                group = tl.arange(0, BLOCK_SIZE // 16)
                qi = seq_index - START
                si = ((qi // 64) * (64 * head_dim // 16) + (group // 4) * 256 + group % 4
                      + (qi % 64 // 16) * 4 + (qi % 16) * 16)
                scale_offset = (batch * num_heads + head) * (seq_len - START) * (head_dim // 16) + si
                tl.store(scales_ptr + scale_offset, sf,
                         (row < ROWS) & (seq_index >= START) & (group < head_dim // 16))


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[int, int, int, int, int]:
    for name, tensor in (("x", x), ("weight", weight), ("cos", cos), ("sin", sin)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")

    if x.ndim != 4:
        raise ValueError(f"x must have shape (batch, seq, heads, head_dim), got {tuple(x.shape)}")
    batch, seq_len, num_heads, head_dim = x.shape
    if min(batch, seq_len, num_heads, head_dim) <= 0:
        raise ValueError(f"x dimensions must all be positive, got {tuple(x.shape)}")
    if weight.shape != (head_dim, ):
        raise ValueError(f"weight must have shape ({head_dim},), got {tuple(weight.shape)}")
    if cos.ndim != 2:
        raise ValueError(f"cos must have shape (seq, rotary_dim), got {tuple(cos.shape)}")
    if sin.shape != cos.shape:
        raise ValueError(f"sin must match cos shape {tuple(cos.shape)}, got {tuple(sin.shape)}")
    if cos.shape[0] != seq_len:
        raise ValueError(f"cos/sin sequence length must be {seq_len}, got {cos.shape[0]}")

    rotary_dim = cos.shape[1]
    if rotary_dim <= 0:
        raise ValueError(f"rotary_dim must be positive, got {rotary_dim}")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim must not exceed head_dim, got rotary_dim={rotary_dim}, head_dim={head_dim}")
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be float16, bfloat16, or float32, got {x.dtype}")
    for name, tensor in (("weight", weight), ("cos", cos), ("sin", sin)):
        if tensor.dtype != x.dtype:
            raise TypeError(f"{name} dtype must match x dtype {x.dtype}, got {tensor.dtype}")
        if tensor.device != x.device:
            raise ValueError(f"{name} device must match x device {x.device}, got {tensor.device}")

    if not isinstance(eps, (float, int)) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be a positive finite number, got {eps!r}")
    return batch, seq_len, num_heads, head_dim, rotary_dim


def _fused_qknorm_rope_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim, rotary_dim = _validate_inputs(x, weight, cos, sin, eps)
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("cos and sin must be contiguous (seq, rotary_dim) tables")
    if not x.is_cuda:
        raise RuntimeError("fused_qknorm_rope requires CUDA tensors")
    if not HAVE_TRITON:
        raise RuntimeError("fused_qknorm_rope requires Triton")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x, weight, cos, sin)):
        raise RuntimeError("fused_qknorm_rope is inference-only and does not implement autograd")

    flat_x = x.reshape(-1, head_dim).contiguous()
    flat_out = torch.empty_like(flat_x)
    block_size = 1 << (head_dim - 1).bit_length()
    rows_per_program = (8 if head_dim == 128 and x.dtype == torch.bfloat16
                        and torch.cuda.get_device_capability(x.device)[0] >= 10 else 1)
    _qknorm_partial_rope_kernel[(triton.cdiv(flat_x.shape[0], rows_per_program), )](
        flat_out,
        flat_x,
        weight,
        cos,
        sin,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        num_heads,
        seq_len,
        eps,
        BLOCK_SIZE=block_size,
        ROWS=flat_x.shape[0],
        ROWS_PER_PROGRAM=rows_per_program,
        num_warps=4,
    )
    return flat_out.view(batch, seq_len, num_heads, head_dim)


@torch.library.custom_op(
    "fastvideo::_minimax_h3_qknorm_rope",
    mutates_args=(),
    device_types="cuda",
)
def _fused_qknorm_rope_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _fused_qknorm_rope_impl(x, weight, cos, sin, eps)


@torch.library.register_fake("fastvideo::_minimax_h3_qknorm_rope")
def _fused_qknorm_rope_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, cos, sin, eps
    return x.new_empty(x.shape)


def fused_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run per-head RMSNorm and partial RoPE in one Sol-Engine-style kernel.

    RMSNorm reduction and RoPE arithmetic stay in FP32 registers until the
    final store. Triton's reduction order and the absence of eager's BF16
    intermediate materializations can produce small, expected rounding drift.

    Row offsets are computed in int64, so inputs beyond 2**31 total elements
    (about 300k tokens per rank at H3's 56 heads x 128 head_dim) address
    correctly.
    """
    if torch.is_grad_enabled() and any(
            isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in (x, weight, cos, sin)):
        raise RuntimeError("fused_qknorm_rope is inference-only and does not implement autograd")
    if torch.compiler.is_compiling():
        return torch.ops.fastvideo._minimax_h3_qknorm_rope(x, weight, cos, sin, eps)
    return _fused_qknorm_rope_impl(x, weight, cos, sin, eps)


__all__ = ["HAVE_TRITON", "fused_qknorm_rope"]


def qknorm_rope_quantize_fp4(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    eps: float, start: int = 0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Keep the BF16 norm/RoPE result and pack its suffix for native Sage FP4.

    The suffix begins on a 128-token boundary. Its codes and 64-row scale
    swizzle are allocated contiguously, as required by the native attention
    kernel. BF16 rounding precedes the per-16-element scale calculation.
    """
    batch, seq_len, heads, dim, rotary_dim = _validate_inputs(x, weight, cos, sin, eps)
    if (not x.is_cuda or x.dtype != torch.bfloat16 or dim != 128 or seq_len % 128
            or start < 0 or start >= seq_len or start % 128):
        raise ValueError("FP4 Q packing requires CUDA BF16 head-128 and aligned sequence/suffix lengths")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("FP4 Q packing requires eager inference")
    x, weight, cos, sin = (t.contiguous() for t in (x, weight, cos, sin))
    out = torch.empty_like(x)
    codes = torch.empty((batch, heads, seq_len - start, 64), device=x.device, dtype=torch.uint8)
    scales = torch.empty((batch, heads, seq_len - start, 8), device=x.device, dtype=torch.float8_e4m3fn)
    _qknorm_partial_rope_kernel[(triton.cdiv(batch * seq_len * heads, 4),)](
        out, x, weight, cos, sin, dim, rotary_dim, rotary_dim // 2, heads, seq_len, eps,
        BLOCK_SIZE=128, ROWS=batch * seq_len * heads, ROWS_PER_PROGRAM=4,
        codes_ptr=codes, scales_ptr=scales, PACK_FP4=True, START=start, num_warps=4,
    )
    return out, (codes, scales)


def gather_qknorm_rope(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float,
    input_index: torch.Tensor, *, fp4_start: int | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Normalize compact projected rows while gathering them into padded tile order.

    Negative indices denote zero projection rows. Pooling and sparse attention
    still receive every original tile row; only the linear projections omit
    padding. An optional aligned suffix is packed directly for native FP4 Q.
    """
    if x.ndim != 4 or input_index.ndim != 1:
        raise ValueError("Gathered normalization requires BSHD input and a one-dimensional row map")
    rows = input_index.numel()
    shape_view = x[:, :1].expand(-1, rows, -1, -1)
    batch, _, heads, dim, rotary_dim = _validate_inputs(shape_view, weight, cos, sin, eps)
    if (not x.is_cuda or x.dtype != torch.bfloat16 or dim != 128 or rows % 128
            or input_index.device != x.device or input_index.dtype not in (torch.int32, torch.int64)
            or not input_index.is_contiguous()):
        raise ValueError("Gathered normalization requires CUDA BF16 head-128 and an aligned integer row map")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("Gathered normalization requires eager inference")
    quantized = fp4_start is not None
    start = 0 if fp4_start is None else fp4_start
    if start < 0 or start >= rows or start % 128:
        raise ValueError("The FP4 query suffix must begin at a valid 128-token boundary")
    x, weight, cos, sin = (t.contiguous() for t in (x, weight, cos, sin))
    out = torch.empty((batch, rows, heads, dim), device=x.device, dtype=x.dtype)
    codes, scales = out, out
    if quantized:
        codes = torch.empty((batch, heads, rows - start, 64), device=x.device, dtype=torch.uint8)
        scales = torch.empty((batch, heads, rows - start, 8), device=x.device, dtype=torch.float8_e4m3fn)
    from .qknorm_rope_vector import qknorm_rope_vector_kernel

    unroll = 16 if quantized else 8
    qknorm_rope_vector_kernel[(triton.cdiv(batch * rows * heads, unroll),)](
        out, x, weight, cos, sin, dim, rotary_dim, rotary_dim // 2, heads, rows, eps,
        BLOCK_SIZE=128, ROWS=batch * rows * heads, ROWS_PER_PROGRAM=unroll,
        codes_ptr=codes, scales_ptr=scales, PACK_FP4=quantized, START=start,
        input_index_ptr=input_index, INPUT_ROWS=x.shape[1], GATHER_INPUT=True, num_warps=4,
    )
    return out, (codes, scales) if quantized else None
