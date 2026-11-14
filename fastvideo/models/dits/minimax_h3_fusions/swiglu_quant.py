# SPDX-License-Identifier: Apache-2.0
"""H3 SwiGLU and exact FlashInfer-compatible NVFP4 activation quantization.

The SwiGLU result is rounded to BF16 before quantization, exactly as in the
separate kernels. Scales use FlashInfer's default fast-math recipe and 128x4
swizzle. Fusing the two passes avoids a full-width BF16 intermediate.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _swiglu_quant(X, SF, Q, S, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < ROWS) & (col < WIDTH)
    value = tl.load(X + row * WIDTH * 2 + col, valid, other=0).to(tl.float32)
    gate = tl.load(X + row * WIDTH * 2 + WIDTH + col, valid, other=0).to(tl.float32)
    y = (value * (gate * tl.sigmoid(gate))).to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(y, (BLOCK // 16, 16))
    amax = tl.max(tl.abs(grouped), 1)
    global_sf = tl.load(SF)
    scale = (global_sf * (amax * _rcp(6.0))).to(tl.float8e4nv)
    inv = tl.where(amax != 0, _rcp(scale.to(tl.float32) * _rcp(global_sf)), 0.0)
    quantized = tl.reshape(grouped * inv[:, None], (BLOCK // 2, 2))
    lo, hi = tl.split(quantized)
    packed = tl.inline_asm_elementwise('''{
        .reg .b8 q;
        cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
        mov.b32 $0, {q,q,q,q};
    }''', constraints='=r,f,f', args=[hi,lo], dtype=tl.uint8, is_pure=True, pack=1)
    qcol = tl.program_id(1) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(Q + row * (WIDTH // 2) + qcol, packed, (row < ROWS) & (qcol < WIDTH // 2))
    group = tl.program_id(1) * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
    sidx = (((((row // 128) * (WIDTH // 64) + group // 4) * 32 + row % 32) * 4
             + (row // 32) % 4) * 4 + group % 4)
    tl.store(S + sidx, scale.to(tl.uint8, bitcast=True), group < WIDTH // 16)



def swiglu_quantize_nvfp4(
    x: torch.Tensor, global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize value-first SwiGLU to packed FP4 codes and padded 128x4 scales.

    Requires contiguous CUDA BF16 inputs with a hidden width divisible by 64,
    a scalar FP32 global scale, and native Blackwell FP4 conversion support.
    The caller selects this only for eager inference with a compatible NVFP4
    output projection; other paths retain the ordinary SwiGLU implementation.
    """
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("NVFP4 SwiGLU requires contiguous CUDA BF16 input")
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] % 128:
        raise ValueError("NVFP4 SwiGLU requires a packed width divisible by 128")
    if global_scale.numel() != 1 or global_scale.dtype != torch.float32 or global_scale.device != x.device:
        raise ValueError("NVFP4 SwiGLU requires a scalar FP32 scale on the input device")
    if torch.is_grad_enabled():
        raise RuntimeError("NVFP4 SwiGLU is inference-only")
    rows, width = x.numel() // x.shape[-1], x.shape[-1] // 2
    quantized = torch.empty((rows, width // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((triton.cdiv(rows, 128) * 128, width // 16), device=x.device, dtype=torch.uint8)
    if rows:
        block = 8192
        _swiglu_quant[(scales.shape[0], triton.cdiv(width, block))](
            x, global_scale, quantized, scales, rows, width, block, num_warps=8,
        )
    return quantized, scales, global_scale
