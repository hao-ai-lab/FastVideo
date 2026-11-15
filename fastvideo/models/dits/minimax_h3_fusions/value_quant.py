# SPDX-License-Identifier: Apache-2.0
"""Share the value read between exact tile pooling and transposed Sage FP4 packing.

For BF16 tile-64/128 pooling, ATen uses four accumulators on each of four warp
rows. Keep that addition order, including the final (0+2)+(1+3) reduction,
instead of a tree sum. Quantization independently reduces the maximum of
16 consecutive tokens and uses Sage's transposed 64-row scale layout.
"""
import torch
import triton
import triton.language as tl

from .swiglu_quant import _rcp


@triton.jit
def _left_four(x, BLOCK: tl.constexpr):
    even, odd = tl.split(tl.reshape(x, (BLOCK, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    return ((a0 + a1) + a2) + a3


@triton.jit
def _pool_and_quantize(X, VALID, POOL, CODES, SCALES,
                       SB: tl.constexpr, SS: tl.constexpr, SH: tl.constexpr,
                       S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, WIDE: tl.constexpr,
                       INPUT_INDEX=None, GATHER_INPUT: tl.constexpr = False, TILE: tl.constexpr = 64):
    tile, batch = tl.program_id(1), tl.program_id(2)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        tile, batch, col = tile.to(tl.int64), batch.to(tl.int64), col.to(tl.int64)
    row = tl.arange(0, 16)
    head, dim = col // D, col % D
    acc = tl.full((16, BLOCK), 0, tl.float32)
    for i in tl.static_range(TILE // 16):
        token = tile * TILE + i * 16 + row
        source = tl.load(INPUT_INDEX + token) if GATHER_INPUT else token
        x = tl.load(X + batch * SB + source[:, None] * SS + head[None, :] * SH + dim[None, :],
                    (col[None, :] < H * D) & (source[:, None] >= 0), other=0).to(tl.float32)
        acc = acc + x
        maximum = tl.max(tl.abs(x), 0)
        sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
        inverse = tl.where(sf.to(tl.float32) == 0, 0, _rcp(sf.to(tl.float32)))
        scaled = x * inverse[None, :]
        pair = tl.trans(tl.reshape(scaled, (8, 2, BLOCK)), (0, 2, 1))
        lo, hi = tl.split(pair)
        codes = tl.inline_asm_elementwise('''{
            .reg .b8 q;
            cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
            mov.b32 $0, {q,q,q,q};
        }''', constraints='=r,f,f', args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
        qrow = tl.arange(0, 8)
        offset = (((batch * H + head[None, :]) * D + dim[None, :]) * (S // 2)
                  + tile * (TILE // 2) + i * 8 + qrow[:, None])
        tl.store(CODES + offset, codes, col[None, :] < H * D)
        group = tile * (TILE // 16) + i
        offset = ((batch * H + head) * D * (S // 16) + (dim // 64) * 64 * (S // 16)
                  + (group // 4) * 256 + group % 4 + (dim % 64 // 16) * 4 + (dim % 16) * 16)
        tl.store(SCALES + offset, sf, col < H * D)
    even, odd = tl.split(tl.reshape(tl.trans(acc), (BLOCK, 4, 2, 2)))
    y0, y2 = tl.split(even)
    y1, y3 = tl.split(odd)
    y0, y1 = _left_four(y0, BLOCK), _left_four(y1, BLOCK)
    y2, y3 = _left_four(y2, BLOCK), _left_four(y3, BLOCK)
    total = (y0 + y2) + (y1 + y3)
    count = tl.load(VALID + tile).to(tl.float32)
    mean = tl.div_rn(total, count)
    tl.store(POOL + ((batch * (S // TILE) + tile) * H * D) + col, mean, col < H * D)


def pool_quantize_values(
    value: torch.Tensor, valid: torch.Tensor, input_index: torch.Tensor | None = None, *, tile_elems: int = 64,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Return FP32 tile means and contiguous native FP4 V codes/scales.

    The caller uses 64- or 128-token tiles, with all logical rows present. Values
    must have unit feature stride and 128-wide heads; token rows and heads
    may be strided. This is an eager, inference-only Blackwell path.
    """
    if (value.ndim != 4 or not value.is_cuda or value.dtype != torch.bfloat16
            or value.shape[-1] != 128 or value.stride(-1) != 1
            or (input_index is None and value.shape[1] % 128)):
        raise ValueError("Value preparation requires CUDA BF16 BSHD, head-128 and padded token rows")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("Value preparation requires eager inference")
    if tile_elems not in (64, 128):
        raise ValueError("Value preparation supports 64- and 128-token tiles")
    batch, seq_len, heads, dim = value.shape
    if input_index is not None:
        if (input_index.ndim != 1 or input_index.numel() % 128 or input_index.device != value.device
                or input_index.dtype not in (torch.int32, torch.int64) or not input_index.is_contiguous()):
            raise ValueError("Compact values require a contiguous aligned integer row map")
        seq_len = input_index.numel()
    if (valid.shape != (seq_len // tile_elems,) or valid.device != value.device or not valid.is_contiguous()
            or valid.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Value preparation requires contiguous integer tile validity counts")
    pooled = torch.empty((batch, seq_len // tile_elems, heads, dim), device=value.device, dtype=torch.float32)
    codes = torch.empty((batch, heads, dim, seq_len // 2), device=value.device, dtype=torch.uint8)
    scales = torch.empty((batch, heads, dim, seq_len // 16), device=value.device, dtype=torch.float8_e4m3fn)
    # Keep wide indexing for large/strided batches without charging small clips
    # for 64-bit vector arithmetic on every packed value.
    span = 1 + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride(), strict=True))
    block = 128 if input_index is not None else 32
    span = max(span, batch * seq_len * heads * dim)
    _pool_and_quantize[(triton.cdiv(heads * dim, block), seq_len // tile_elems, batch)](
        value, valid, pooled, codes, scales, *value.stride()[:3], seq_len, heads, dim, block, span >= 2**31,
        INPUT_INDEX=input_index, GATHER_INPUT=input_index is not None, TILE=tile_elems, num_warps=4,
    )
    return pooled.permute(0, 2, 1, 3), (codes, scales)
