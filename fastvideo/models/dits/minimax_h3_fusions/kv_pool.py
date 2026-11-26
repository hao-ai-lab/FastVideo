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
                 VP=None, V4=None, VS4=None, HEAD_FIRST: tl.constexpr = False,
                 K4D=None, S4D=None, EMIT_BF16: tl.constexpr = True,
                 EMIT_BF16_V: tl.constexpr = True, DENSE_K: tl.constexpr = False):
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
        k8v = (key8 * scale8[:, None]).to(tl.bfloat16)
        if EMIT_BF16:
            tl.store(K8 + offset, k8v, valid)
        # Emit the dense-prefix FP4 K (codes + 128x4-swizzled e4m3 scales) directly
        # so the sm_12x dense kernel no longer re-reads and re-quantizes K8. The
        # quantizer here is bit-identical to fp4quant_cuda.scaled_fp4_quant_permute
        # on the same bf16 input (per-16-element e2m1/e4m3, 32-row permuted store).
        if DENSE_K:
            k8b = k8v.to(tl.float32)
            groups8 = tl.reshape(k8b, (R, 8, 16))
            maximum8 = tl.max(tl.abs(groups8), 2)
            sf8 = (maximum8 * (1.0 / 6.0)).to(tl.float8e4nv)
            inv8b = tl.where(sf8.to(tl.float32) == 0, 0, _rcp(sf8.to(tl.float32)))
            q8 = tl.reshape(groups8 * inv8b[:, :, None], (R, 64, 2))
            lo8, hi8 = tl.split(q8)
            packed8 = tl.inline_asm_elementwise('''{
                .reg .b8 q;
                cvt.rn.satfinite.e2m1x2.f32 q,$1,$2;
                mov.b32 $0,{q,q,q,q};
            }''', constraints='=r,f,f', args=[hi8, lo8], dtype=tl.uint8, is_pure=True, pack=1)
            dsafe = tl.where(destination >= 0, destination, 0)
            drem = dsafe % 32
            token = dsafe // 32 * 32 + (drem // 8) * 2 + (drem % 8 // 2) * 8 + drem % 2
            c4d = tl.arange(0, 64)
            tl.store(K4D + ((batch * H + head) * VALID + token[:, None]) * 64 + c4d[None, :],
                     packed8, valid)
            g8 = tl.arange(0, 8)
            si8 = ((token[:, None] // 64) * 512 + (g8[None, :] // 4) * 256 + g8[None, :] % 4
                   + (token[:, None] % 64 // 16) * 4 + (token[:, None] % 16) * 16)
            tl.store(S4D + (batch * H + head) * VALID * 8 + si8, sf8, valid)
        value_source = tl.load(VALUE_INDEX + source, row < N, other=-1) if GATHER_VALUE else source
        value = tl.load(V + batch * VB + value_source[:, None] * VS + head * VH + col[None, :],
                        mask & (value_source[:, None] >= 0), other=0)
        if EMIT_BF16_V:
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


@triton.jit
def _dense_v_pack(V, VALID, V4D, VS4D, VB: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
                  H: tl.constexpr, N: tl.constexpr, N_VALID: tl.constexpr,
                  GROUP: tl.constexpr = 16, VALUE_INDEX=None, GATHER_VALUE: tl.constexpr = False):
    """Gather-quantize V into the dense-prefix transposed FP4 layout.

    Program ``dg`` handles the ``GROUP`` destination tokens ``[dg*GROUP, (dg+1)*GROUP)``:
    it gathers their source rows through ``VALID`` (destination -> source), quantizes
    with the same per-16-token e4m3 scale / e2m1 codes as
    ``fp4quant_cuda.scaled_fp4_quant_trans`` on the destination-ordered BF16 V, and
    stores into the ``[B, H, 128, N/2]`` codes / ``[B, H, 128, N/16]`` scale layout.
    Tokens past ``N_VALID`` read as zero, matching the zero tail of that BF16 V.
    """
    dg = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    d = tl.arange(0, 128)
    i = tl.arange(0, GROUP)
    tok = dg * GROUP + i
    row = tl.load(VALID + tok, tok < N_VALID, other=-1).to(tl.int64)
    if GATHER_VALUE:
        vsrc = tl.load(VALUE_INDEX + row, (tok < N_VALID) & (row >= 0), other=-1).to(tl.int64)
    else:
        vsrc = row
    valid = (tok < N_VALID) & (vsrc >= 0)
    v = tl.load(V + batch * VB + vsrc[:, None] * VS + head * VH + d[None, :],
                valid[:, None], other=0).to(tl.float32)
    maximum = tl.max(tl.abs(v), 0)
    sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
    inv = tl.where(sf.to(tl.float32) == 0, 0, _rcp(sf.to(tl.float32)))
    scaled = v * inv[None, :]
    grp = tl.reshape(scaled, (GROUP // 2, 2, 128))
    vlo, vhi = tl.split(tl.trans(grp, (0, 2, 1)))
    codes = tl.inline_asm_elementwise("""{
        .reg .b8 q;
        cvt.rn.satfinite.e2m1x2.f32 q,$1,$2;
        mov.b32 $0,{q,q,q,q};
    }""", constraints='=r,f,f', args=[vhi, vlo], dtype=tl.uint8, is_pure=True, pack=1)
    j = tl.arange(0, GROUP // 2)
    voffset = ((batch * H + head) * 128 + d[None, :]) * (N // 2) + dg * (GROUP // 2) + j[:, None]
    tl.store(V4D + voffset, codes)
    soffset = ((batch * H + head) * 128 * (N // 16)
               + (d // 64) * 64 * (N // 16)
               + (dg // 4) * 256 + dg % 4
               + (d % 64 // 16) * 4 + (d % 16) * 16)
    tl.store(VS4D + soffset, sf)


def pool_quantize_prefix_kv(
    k: torch.Tensor, v: torch.Tensor, mean: torch.Tensor, inverse: torch.Tensor,
    n_valid: int, valid: torch.Tensor, value_index: torch.Tensor | None, *, tile_elems: int = 64,
    dense_k: bool = False, emit_bf16_k: bool = True,
    dense_v: bool = False, emit_bf16_v: bool = True, valid_index: torch.Tensor | None = None,
) -> tuple[
    tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None, torch.Tensor | None, torch.Tensor,
          tuple[torch.Tensor, torch.Tensor] | None],
    tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None],
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
    for index, size in ((inverse, rows), (value_index, rows), (valid, rows // tile_elems),
                        (valid_index, n_valid)):
        if index is None:
            continue
        if (index.shape != (size,) or index.device != k.device or not index.is_contiguous()
                or index.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Combined KV preparation requires contiguous integer row maps and tile counts")
    if dense_v and valid_index is None:
        raise ValueError("Dense-prefix value packing requires a destination-to-source row map")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("Combined KV preparation requires eager inference")
    k4 = torch.empty((batch, heads, rows, 64), device=k.device, dtype=torch.uint8)
    s4 = torch.empty((batch, heads, rows, 8), device=k.device, dtype=torch.float8_e4m3fn)
    # The dense-prefix K/V are padded with zero rows to the FP4 kernel's 128-row block; the kernel
    # masks keys past n_valid, and the SDPA fallback trims them.
    padded = ((n_valid + 127) // 128) * 128
    if emit_bf16_v:
        v8_buf = torch.empty((batch, heads, padded, 128), device=k.device, dtype=torch.bfloat16)
        if padded > n_valid:
            v8_buf[:, :, n_valid:].zero_()
    else:
        v8_buf = torch.empty((1, ), device=k.device, dtype=torch.bfloat16)  # Never stored to (EMIT_BF16_V=False).
    k4d = s4d = None
    if dense_k:
        # Zero only the swizzled tail. Both dense layouts permute within a fixed token block (32 rows
        # for the codes, 64 for the 128x4-swizzled scales), so tail destinations of a partial block
        # can sit below n_valid: zero whole blocks from the first one that can hold a tail slot.
        k4d = torch.empty((batch, heads, padded, 64), device=k.device, dtype=torch.uint8)
        s4d = torch.empty((batch, heads, padded, 8), device=k.device, dtype=torch.float8_e4m3fn)
        if padded > n_valid:
            k4d[:, :, (n_valid // 32) * 32:].zero_()
            s4d[:, :, (n_valid // 64) * 64:].zero_()
    if emit_bf16_k:
        k8_buf = torch.empty((batch, heads, padded, 128), device=k.device, dtype=torch.bfloat16)
        if padded > n_valid:
            k8_buf[:, :, n_valid:].zero_()
        k8 = k8_buf.transpose(1, 2)
        k8_arg: torch.Tensor = k8
    else:
        k8 = None
        k8_arg = torch.empty((1,), device=k.device, dtype=torch.bfloat16)
    if emit_bf16_v:
        v8 = v8_buf.transpose(1, 2)
    else:
        v8 = None
    kp = torch.empty((batch, rows // tile_elems, heads, 128), device=k.device, dtype=torch.float32)
    vp = torch.empty_like(kp)
    v4 = torch.empty((batch, heads, 128, rows // 2), device=k.device, dtype=torch.uint8)
    vs4 = torch.empty((batch, heads, 128, rows // 16), device=k.device, dtype=torch.float8_e4m3fn)
    _smooth_pack[(rows // tile_elems, batch, heads)](
        k, v, mean, inverse, k4, s4, k8_arg, v8 if v8 is not None else v8_buf, *k.stride()[:3], *v.stride()[:3],
        batch, rows, heads, padded, tile_elems, VALUE_INDEX=value_index, GATHER_VALUE=value_index is not None,
        POOL=kp, TILE_VALID=valid, POOL_KEYS=True, VP=vp, V4=v4, VS4=vs4, HEAD_FIRST=True,
        K4D=k4d, S4D=s4d, EMIT_BF16=emit_bf16_k, EMIT_BF16_V=emit_bf16_v, DENSE_K=dense_k,
        num_warps=4 if tile_elems == 64 else 8,
    )
    v4d = vs4d = None
    if dense_v:
        assert valid_index is not None
        v4d = torch.empty((batch, heads, 128, padded // 2), device=k.device, dtype=torch.uint8)
        vs4d = torch.empty((batch, heads, 128, padded // 16), device=k.device,
                           dtype=torch.float8_e4m3fn)
        _dense_v_pack[(padded // 16, batch, heads)](
            v, valid_index, v4d, vs4d, *v.stride()[:3], heads, padded, n_valid,
            VALUE_INDEX=value_index, GATHER_VALUE=value_index is not None, num_warps=4,
        )
    return (((k4, s4), k8, v8, kp.permute(0, 2, 1, 3), (k4d, s4d) if dense_k else None),
            (vp.permute(0, 2, 1, 3), (v4, vs4), (v4d, vs4d) if dense_v else None))
