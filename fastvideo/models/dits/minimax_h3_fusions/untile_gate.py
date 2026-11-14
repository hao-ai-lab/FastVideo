# SPDX-License-Identifier: Apache-2.0
"""Fuse VSA's packed-order gather and compression-gate addcmul."""

import torch
import triton
import triton.language as tl

from .swiglu_quant import _rcp as rcp


@triton.jit
def _untile_gate_kernel(O, P, C, G, Untile, RowTile, Y, Q, S, SF,
                        OB: tl.constexpr, OH: tl.constexpr, OR: tl.constexpr,
                        PB: tl.constexpr, PH: tl.constexpr, PR: tl.constexpr,
                        CB: tl.constexpr, CT: tl.constexpr, CH: tl.constexpr,
                        PREFIX: tl.constexpr, START: tl.constexpr,
                        ROWS: tl.constexpr, HD: tl.constexpr, D: tl.constexpr,
                        BLOCK: tl.constexpr, QUANTIZED: tl.constexpr):
    row = tl.program_id(0)
    batch = tl.program_id(2)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head, dim = col // D, col % D
    src = tl.load(Untile + row).to(tl.int64)
    tile = tl.load(RowTile + row).to(tl.int64)
    if PREFIX > 0:
        if src < PREFIX:
            a = tl.load(P + batch * PB + head * PH + src * PR + dim, col < HD, other=0).to(tl.float32)
        else:
            a = tl.load(O + batch * OB + head * OH + (src - START) * OR + dim, col < HD, other=0).to(tl.float32)
    else:
        a = tl.load(O + batch * OB + head * OH + src * OR + dim, col < HD, other=0).to(tl.float32)
    b = tl.load(C + batch * CB + tile * CT + head * CH + dim, col < HD, other=0).to(tl.float32)
    offset = (batch * ROWS + row).to(tl.int64) * HD + col
    gate = tl.load(G + offset, col < HD, other=0).to(tl.float32)
    # ATen's BF16 addcmul uses float32 fused multiply-add and one final cast.
    if QUANTIZED:
        y = tl.where(col < HD, tl.fma(b, gate, a).to(tl.bfloat16).to(tl.float32), 0.0)
        grouped = tl.reshape(y, (BLOCK // 16, 16))
        amax = tl.max(tl.abs(grouped), 1)
        global_sf = tl.load(SF)
        scale = (global_sf * (amax * rcp(6.0))).to(tl.float8e4nv)
        inv = tl.where(amax != 0, rcp(scale.to(tl.float32) * rcp(global_sf)), 0.0)
        quant = tl.reshape(grouped * inv[:, None], (BLOCK // 2, 2))
        lo, hi = tl.split(quant)
        packed = tl.inline_asm_elementwise('''{
            .reg .b8 q;
            cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
            mov.b32 $0, {q,q,q,q};
        }''', constraints="=r,f,f", args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
        packed_row = (batch * ROWS + row).to(tl.int64)
        qc = tl.program_id(1) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
        tl.store(Q + packed_row * (HD // 2) + qc, packed, qc < HD // 2)
        group = tl.program_id(1) * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
        scale_tile = (packed_row // 128) * (HD // 64) + group // 4
        si = ((scale_tile * 32 + packed_row % 32) * 4 + (packed_row // 32) % 4) * 4 + group % 4
        tl.store(S + si, scale.to(tl.uint8, bitcast=True), group < HD // 16)
    else:
        tl.store(Y + offset, tl.fma(b, gate, a), col < HD)


def untile_gate(out: torch.Tensor | tuple[torch.Tensor, torch.Tensor, int], compressed: torch.Tensor,
                gate: torch.Tensor, untile: torch.Tensor, row_tile: torch.Tensor,
                global_scale: torch.Tensor | None = None,
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine attention and compression in packed order, optionally preparing NVFP4 directly."""
    batch, rows, hd = gate.shape
    quantized = global_scale is not None
    if quantized:
        if gate.dtype != torch.bfloat16 or hd % 64:
            raise ValueError("Fused gate quantization requires BF16 and a width divisible by 64")
        if (global_scale.numel() != 1 or global_scale.dtype != torch.float32
                or global_scale.device != gate.device):
            raise ValueError("Fused gate quantization requires a scalar FP32 scale on the input device")
        codes = torch.empty((batch * rows, hd // 2), device=gate.device, dtype=torch.uint8)
        scales = torch.zeros((triton.cdiv(batch * rows, 128) * 128, hd // 16),
                             device=gate.device, dtype=torch.uint8)
        result = gate  # Unused output pointer in the quantized specialization.
    else:
        result = torch.empty_like(gate)
        codes, scales, global_scale = result, result, result
    if isinstance(out, tuple):
        prefix, out, start = out
        prefix_rows = prefix.shape[2]
    else:
        prefix, prefix_rows, start = out, 0, 0
    _untile_gate_kernel[(rows, triton.cdiv(hd, 512), batch)](
        out, prefix, compressed, gate, untile, row_tile, result, codes, scales, global_scale,
        *out.stride()[:3], *prefix.stride()[:3], *compressed.stride()[:3],
        prefix_rows, start, rows, hd, out.shape[-1], 512, quantized,
    )
    return (codes, scales, global_scale) if quantized else result
