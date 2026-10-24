# SPDX-License-Identifier: Apache-2.0
"""Share the compact value read across prefix attention, pooling and native FP4 packing."""
import torch
import triton
import triton.language as tl

from .swiglu_quant import _rcp
from .tile_pool import _tile_sum


@triton.jit
def _smooth_pack(K, V, MEAN, INV, K4, S4, K8, V8, KB: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
                 VB: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
                 B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, VALID: tl.constexpr,
                 R: tl.constexpr, VALUE_INDEX=None, GATHER_VALUE: tl.constexpr = False,
                 PREPARE_PREFIX: tl.constexpr = True,
                 POOL=None, TILE_VALID=None, POOL_KEYS: tl.constexpr = False,
                 VP=None, V4=None, VS4=None, HEAD_FIRST: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    col = tl.arange(0, 128)
    rem = row % 32
    permuted = row // 32 * 32 + (rem // 8) * 2 + (rem % 8 // 2) * 8 + rem % 2
    source = row if POOL_KEYS else permuted
    destination4 = permuted if POOL_KEYS else row
    mask = row[:, None] < N
    key = tl.load(K + batch * KB + source[:, None] * KS + head * KH + col[None, :],
                  mask, other=0).to(tl.float32)
    if POOL_KEYS:
        total = _tile_sum(key, R)
        tile = tl.program_id(0)
        count = tl.load(TILE_VALID + tile).to(tl.float32)
        tl.store(POOL + ((batch * (N // R) + tile) * H + head) * 128 + col, tl.div_rn(total, count))
    mean = tl.load(MEAN + (batch * H + head) * 128 + col).to(tl.float32)
    key = (key - mean[None, :]).to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(key, (R, 8, 16))
    maximum = tl.max(tl.abs(groups), 2)
    sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
    inv4 = tl.where(sf.to(tl.float32) == 0, 0, _rcp(sf.to(tl.float32)))
    q = tl.reshape(groups * inv4[:, :, None], (R, 64, 2))
    lo, hi = tl.split(q)
    packed = tl.inline_asm_elementwise('''{
        .reg .b8 q;
        cvt.rn.satfinite.e2m1x2.f32 q,$1,$2;
        mov.b32 $0,{q,q,q,q};
    }''', constraints='=r,f,f', args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
    c4 = tl.arange(0, 64)
    tl.store(K4 + ((batch * H + head) * N + destination4[:, None]) * 64 + c4[None, :], packed, mask)
    g = tl.arange(0, 8)
    si = ((destination4[:, None] // 64) * 512 + (g[None, :] // 4) * 256 + g[None, :] % 4
          + (destination4[:, None] % 64 // 16) * 4 + (destination4[:, None] % 16) * 16)
    tl.store(S4 + (batch * H + head) * N * 8 + si, sf, mask)
    if PREPARE_PREFIX:
        destination = tl.load(INV + source, row < N, other=-1).to(tl.int64)
        valid = mask & (destination[:, None] >= 0)
        scale8 = tl.max(tl.abs(key), 1) / 448.0
        inv8 = tl.where(scale8 == 0, 0, 1.0 / scale8)
        key8 = tl.minimum(tl.maximum(key * inv8[:, None], -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)
        offset = ((batch * VALID + destination[:, None]) * H + head) * 128 + col[None, :]
        if HEAD_FIRST:
            offset = ((batch * H + head) * VALID + destination[:, None]) * 128 + col[None, :]
        tl.store(K8 + offset, key8 * scale8[:, None], valid)
        value_source = tl.load(VALUE_INDEX + source, row < N, other=-1) if GATHER_VALUE else source
        value = tl.load(V + batch * VB + value_source[:, None] * VS + head * VH + col[None, :],
                        mask & (value_source[:, None] >= 0), other=0)
        tl.store(V8 + offset, value, valid)


    # V was already gathered for dense prefix attention; keep its rounded bits.
    vf = value.to(tl.float32)
    vg = tl.reshape(vf, (R // 16, 16, 128))
    maximum_v = tl.max(tl.abs(vg), 1)
    sfv = (maximum_v * (1.0 / 6.0)).to(tl.float8e4nv)
    invv = tl.where(sfv.to(tl.float32) == 0, 0, _rcp(sfv.to(tl.float32)))
    scaled_v = tl.reshape(vg * invv[:, None, :], (R // 2, 2, 128))
    vlo, vhi = tl.split(tl.trans(scaled_v, (0, 2, 1)))
    vcodes = tl.inline_asm_elementwise("""{
        .reg .b8 q;
        cvt.rn.satfinite.e2m1x2.f32 q,$1,$2;
        mov.b32 $0,{q,q,q,q};
    }""", constraints='=r,f,f', args=[vhi, vlo], dtype=tl.uint8, is_pure=True, pack=1)
    vr = tl.arange(0, R // 2)
    voffset = (((batch * H + head) * 128 + col[None, :]) * (N // 2)
               + tile * (R // 2) + vr[:, None])
    tl.store(V4 + voffset, vcodes)
    group_v = tile * (R // 16) + tl.arange(0, R // 16)
    soffset = ((batch * H + head) * 128 * (N // 16)
               + (col[None, :] // 64) * 64 * (N // 16)
               + (group_v[:, None] // 4) * 256 + group_v[:, None] % 4
               + (col[None, :] % 64 // 16) * 4 + (col[None, :] % 16) * 16)
    tl.store(VS4 + soffset, sfv)
    vtotal = _tile_sum(vf, R)
    tl.store(VP + ((batch * (N // R) + tile) * H + head) * 128 + col, tl.div_rn(vtotal, count))


def pool_quantize_prefix_kv(
    k: torch.Tensor, v: torch.Tensor, mean: torch.Tensor, inverse: torch.Tensor,
    n_valid: int, valid: torch.Tensor, value_index: torch.Tensor | None, *, tile_elems: int = 64,
) -> tuple[
    tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
]:
    """Return exact pooled/native KV plus dense-prefix KV with contiguous head sequences.

    Keys are already normalized in tile order. Values remain in compact projection
    order with a row map, or padded projection order without one. A row map
    zero-fills missing source rows; direct inputs retain every padded value. Each value is read
    once for the prefix BF16 tensor, FP32 tile mean, and transposed FP4 codes/scales.
    """
    if (not k.is_cuda or k.ndim != 4 or v.ndim != 4 or k.dtype != torch.bfloat16
            or v.dtype != k.dtype or v.device != k.device or k.shape[0] != v.shape[0]
            or k.shape[2:] != v.shape[2:] or k.shape[-1] != 128 or k.shape[1] % 128
            or tile_elems not in (64, 128) or (value_index is None and v.shape[1] != k.shape[1])
            or k.stride(-1) != 1 or v.stride(-1) != 1):
        raise ValueError("Combined KV preparation requires CUDA BF16 BSHD with aligned head-128 keys")
    batch, rows, heads, _ = k.shape
    if (mean.shape != (batch, 1, heads, 128) or mean.dtype != k.dtype or mean.device != k.device
            or not mean.is_contiguous() or not 0 < n_valid <= rows):
        raise ValueError("Combined KV preparation requires a BF16 key mean and valid prefix extent")
    for index, size in ((inverse, rows), (value_index, rows), (valid, rows // tile_elems)):
        if index is None:
            continue
        if (index.shape != (size,) or index.device != k.device or not index.is_contiguous()
                or index.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Combined KV preparation requires contiguous integer row maps and tile counts")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("Combined KV preparation requires eager inference")
    k4 = torch.empty((batch, heads, rows, 64), device=k.device, dtype=torch.uint8)
    s4 = torch.empty((batch, heads, rows, 8), device=k.device, dtype=torch.float8_e4m3fn)
    k8 = torch.empty((batch, heads, n_valid, 128), device=k.device, dtype=torch.bfloat16).transpose(1, 2)
    v8 = torch.empty_like(k8)
    kp = torch.empty((batch, rows // tile_elems, heads, 128), device=k.device, dtype=torch.float32)
    vp = torch.empty_like(kp)
    v4 = torch.empty((batch, heads, 128, rows // 2), device=k.device, dtype=torch.uint8)
    vs4 = torch.empty((batch, heads, 128, rows // 16), device=k.device, dtype=torch.float8_e4m3fn)
    _smooth_pack[(rows // tile_elems, batch, heads)](
        k, v, mean, inverse, k4, s4, k8, v8, *k.stride()[:3], *v.stride()[:3],
        batch, rows, heads, n_valid, tile_elems, VALUE_INDEX=value_index, GATHER_VALUE=value_index is not None,
        POOL=kp, TILE_VALID=valid, POOL_KEYS=True, VP=vp, V4=v4, VS4=vs4, HEAD_FIRST=True, num_warps=4 if tile_elems == 64 else 8,
    )
    return ((k4, s4), k8, v8, kp.permute(0, 2, 1, 3)), (vp.permute(0, 2, 1, 3), (v4, vs4))
