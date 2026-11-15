# SPDX-License-Identifier: Apache-2.0
"""Fuse BF16 key smoothing, Sage FP4 packing, and valid prefix K/V preparation.

The FP4 token permutation and 64-row scale swizzle match fastvideo-kernel's
scaled_fp4_quant_permute. Both FP4 and FP8 consume the rounded BF16 key.
"""
import torch
import triton
import triton.language as tl

from fastvideo.distributed.async_collectives import PeerSlices

from .swiglu_quant import _rcp
from .value_quant import _left_four


@triton.jit
def _smooth_pack(K, V, MEAN, INV, K4, S4, K8, V8, KB: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
                 VB: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
                 B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, VALID: tl.constexpr,
                 R: tl.constexpr, VALUE_INDEX=None, GATHER_VALUE: tl.constexpr = False,
                 PREPARE_PREFIX: tl.constexpr = True, COPY_VALUES: tl.constexpr = True,
                 POOL=None, TILE_VALID=None, POOL_KEYS: tl.constexpr = False,
                 PACKED_SCALE=None, TILE_INVERSE=None, Remote=None, LOCAL_ROWS: tl.constexpr = 0,
                 PAIR: tl.constexpr = False, RANK: tl.constexpr = 0, RECEIVED_KEYS: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    col = tl.arange(0, 128)
    rem = row % 32
    permuted = row // 32 * 32 + (rem // 8) * 2 + (rem % 8 // 2) * 8 + rem % 2
    source = row if POOL_KEYS else permuted
    destination4 = permuted if POOL_KEYS else row
    mask = row[:, None] < N
    if RECEIVED_KEYS:
        packed_row = tl.load(TILE_INVERSE + source, row < N, other=-1)
        valid_key = mask & (packed_row[:, None] >= 0)
        peer = packed_row // LOCAL_ROWS
        packed = (peer * 3 + 1) * LOCAL_ROWS + packed_row % LOCAL_ROWS
        if PAIR:
            offset = (LOCAL_ROWS + packed_row % LOCAL_ROWS)[:, None] * H * 128 + head * 128 + col[None, :]
            pointer = tl.where(peer[:, None] == RANK, K + offset, Remote + offset)
        else:
            pointer = K + packed[:, None] * H * 128 + head * 128 + col[None, :]
        code = tl.load(pointer, valid_key, other=0.0).to(tl.float32)
        transport_scale = tl.load(PACKED_SCALE + packed * H + head, (row < N) & (packed_row >= 0), other=0)
        key = (code * transport_scale[:, None]).to(tl.bfloat16).to(tl.float32)
    else:
        key = tl.load(K + batch * KB + source[:, None] * KS + head * KH + col[None, :],
                      mask, other=0).to(tl.float32)
    if POOL_KEYS:
        raw = tl.trans(tl.reshape(key, (R // 16, 16, 128)), (1, 2, 0))
        if R == 64:
            even, odd = tl.split(tl.reshape(raw, (16, 128, 2, 2)))
            x0, x2 = tl.split(even)
            x1, x3 = tl.split(odd)
            acc = ((x0 + x1) + x2) + x3
        else:
            even, odd = tl.split(tl.reshape(raw, (16, 128, 4, 2)))
            x04, x26 = tl.split(tl.reshape(even, (16, 128, 2, 2)))
            x15, x37 = tl.split(tl.reshape(odd, (16, 128, 2, 2)))
            x0, x4 = tl.split(x04)
            x1, x5 = tl.split(x15)
            x2, x6 = tl.split(x26)
            x3, x7 = tl.split(x37)
            acc = ((((((x0 + x1) + x2) + x3) + x4) + x5) + x6) + x7
        even, odd = tl.split(tl.reshape(tl.trans(acc), (128, 4, 2, 2)))
        y0, y2 = tl.split(even)
        y1, y3 = tl.split(odd)
        y0, y1 = _left_four(y0, 128), _left_four(y1, 128)
        y2, y3 = _left_four(y2, 128), _left_four(y3, 128)
        total = (y0 + y2) + (y1 + y3)
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
        tl.store(K8 + offset, key8 * scale8[:, None], valid)
        if COPY_VALUES:
            value_source = tl.load(VALUE_INDEX + source, row < N, other=-1) if GATHER_VALUE else source
            value = tl.load(V + batch * VB + value_source[:, None] * VS + head * VH + col[None, :], valid, other=0)
            tl.store(V8 + offset, value, valid)


def _prepare_keys(
    k: torch.Tensor, v: torch.Tensor, mean: torch.Tensor, inverse: torch.Tensor, n_valid: int,
    value_index: torch.Tensor | None = None, pool_valid: torch.Tensor | None = None, *, tile_elems: int = 64,
    prefix_value: torch.Tensor | None = None,
    received_keys: tuple[torch.Tensor | PeerSlices, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return Sage-packed K and FP8-rounded K/BF16 V gathered into prefix order."""
    if (not k.is_cuda or k.dtype != torch.bfloat16 or v.dtype != k.dtype or v.device != k.device
            or k.ndim != 4 or v.ndim != 4 or k.shape[0] != v.shape[0] or k.shape[2:] != v.shape[2:]
            or (value_index is None and k.shape != v.shape) or k.shape[-1] != 128 or k.shape[1] % 128
            or k.stride(-1) != 1 or v.stride(-1) != 1):
        raise ValueError("Key quantization requires CUDA BF16 BSHD with 128-wide heads and padded token rows")
    if (mean.shape != (k.shape[0], 1, k.shape[2], 128) or mean.dtype != k.dtype
            or mean.device != k.device or not mean.is_contiguous()):
        raise ValueError("Key quantization requires a contiguous BF16 token mean")
    if (inverse.shape != (k.shape[1],) or inverse.device != k.device
            or inverse.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Key quantization requires an integer inverse row map on the input device")
    if value_index is not None and (value_index.shape != (k.shape[1],) or value_index.device != v.device
                                    or value_index.dtype not in (torch.int32, torch.int64)
                                    or not value_index.is_contiguous()):
        raise ValueError("Compact prefix values require an integer tile-to-packed row map")
    if torch.is_grad_enabled():
        raise RuntimeError("Key quantization is inference-only")
    batch, rows, heads, _ = k.shape
    k4 = torch.empty((batch, heads, rows, 64), device=k.device, dtype=torch.uint8)
    s4 = torch.empty((batch, heads, rows, 8), device=k.device, dtype=torch.float8_e4m3fn)
    k8 = torch.empty((batch, n_valid, heads, 128), device=k.device, dtype=torch.bfloat16)
    if prefix_value is not None and (prefix_value.shape != k8.shape or prefix_value.dtype != k8.dtype
                                     or prefix_value.device != k8.device or not prefix_value.is_contiguous()):
        raise ValueError("Prepared prefix values must match the contiguous BF16 prefix-key output")
    v8 = torch.empty_like(k8) if prefix_value is None else prefix_value
    pooled = None
    if pool_valid is not None:
        if tile_elems not in (64, 128):
            raise ValueError("Key pooling supports 64- and 128-token tiles")
        if (pool_valid.shape != (rows // tile_elems,) or pool_valid.device != k.device
                or pool_valid.dtype not in (torch.int32, torch.int64) or not pool_valid.is_contiguous()):
            raise ValueError("Key pooling requires contiguous integer tile validity counts")
        pooled = torch.empty((batch, rows // tile_elems, heads, 128), device=k.device, dtype=torch.float32)
    key_input, remote = k, k
    transport_scale, tile_inverse, local_rows, rank, pair = None, None, 0, 0, None
    if received_keys is not None:
        payload, transport_scale, tile_inverse = received_keys
        pair = payload if isinstance(payload, PeerSlices) else None
        template = pair.template if pair is not None else payload
        if (batch != 1 or pool_valid is not None or prefix_value is None
                or template.ndim != 5 or template.shape[1] != 3 or template.shape[2] < 1
                or template.shape[3:] != (heads, 128) or template.dtype != torch.float8_e4m3fn
                or template.device != k.device or not template.is_contiguous()
                or transport_scale.shape != template.shape[:-1] or transport_scale.dtype != torch.float32
                or transport_scale.device != k.device or not transport_scale.is_contiguous()
                or tile_inverse.shape != (rows,) or tile_inverse.dtype not in (torch.int32, torch.int64)
                or tile_inverse.device != k.device or not tile_inverse.is_contiguous()):
            raise ValueError("Received keys require FP8 QKV, FP32 scales, an inverse row map and prepared prefix values")
        if pair is not None and (template.shape[0] != 2 or pair.rank not in (0, 1)
                or any(t.shape != template.shape[1:] or t.dtype != template.dtype
                       or t.device != template.device or not t.is_contiguous() for t in (pair.local, pair.remote))):
            raise ValueError("Direct-local received keys require matching contiguous two-rank peer slices")
        key_input = pair.local if pair is not None else template
        remote = pair.remote if pair is not None else template
        local_rows, rank = template.shape[2], pair.rank if pair is not None else 0
    block = tile_elems if pooled is not None else 4
    _smooth_pack[(triton.cdiv(rows, block), batch, heads)](
        key_input, v, mean, inverse, k4, s4, k8, v8, *k.stride()[:3], *v.stride()[:3],
        batch, rows, heads, n_valid, block, VALUE_INDEX=value_index, GATHER_VALUE=value_index is not None,
        POOL=pooled, TILE_VALID=pool_valid, POOL_KEYS=pooled is not None, COPY_VALUES=prefix_value is None,
        PACKED_SCALE=transport_scale, TILE_INVERSE=tile_inverse, Remote=remote, LOCAL_ROWS=local_rows,
        PAIR=pair is not None, RANK=rank, RECEIVED_KEYS=received_keys is not None,
        num_warps=8 if pooled is not None else 4,
    )
    return (k4, s4), k8, v8, pooled


def quantize_smoothed_keys(
    k: torch.Tensor, v: torch.Tensor, mean: torch.Tensor, inverse: torch.Tensor, n_valid: int,
    value_index: torch.Tensor | None = None, *, prefix_value: torch.Tensor | None = None,
    received_keys: tuple[torch.Tensor | PeerSlices, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Prepare keys and optionally consume values already in dense-prefix order.

    When ``prefix_value`` is supplied, V is used only for metadata validation;
    its contents are not read. The supplied buffer is returned without a copy.
    With ``received_keys=(payload, scale, tile_inverse)``, K is metadata only;
    keys are expanded from the current call's FP8 packets with the same BF16
    rounding before smoothing. This requires prepared prefix values and batch one.
    """
    packed, prefix_key, prefix_value, _ = _prepare_keys(
        k, v, mean, inverse, n_valid, value_index, prefix_value=prefix_value, received_keys=received_keys,
    )
    return packed, prefix_key, prefix_value


def pool_quantize_smoothed_keys(
    k: torch.Tensor, v: torch.Tensor, mean: torch.Tensor, inverse: torch.Tensor, n_valid: int,
    valid: torch.Tensor, value_index: torch.Tensor | None = None, *, tile_elems: int = 64,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Also return exact ATen-order FP32 means of unsmoothed 64- or 128-token tiles."""
    packed, prefix_key, prefix_value, pooled = _prepare_keys(
        k, v, mean, inverse, n_valid, value_index, valid, tile_elems=tile_elems,
    )
    assert pooled is not None
    return packed, prefix_key, prefix_value, pooled.permute(0, 2, 1, 3)


def quantize_smoothed_keys_only(k: torch.Tensor, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the rounded BF16 key shift without allocating dense-prefix outputs."""
    if (not k.is_cuda or k.dtype != torch.bfloat16 or k.ndim != 4 or k.shape[-1] != 128
            or k.shape[1] % 128 or k.stride(-1) != 1):
        raise ValueError("Key-only preparation requires CUDA BF16 head-128 with padded token rows")
    if (mean.shape != (k.shape[0], 1, k.shape[2], 128) or mean.dtype != k.dtype
            or mean.device != k.device or not mean.is_contiguous()):
        raise ValueError("Key-only preparation requires a contiguous BF16 token mean")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("Key-only preparation requires eager inference")
    batch, rows, heads, _ = k.shape
    codes = torch.empty((batch, heads, rows, 64), device=k.device, dtype=torch.uint8)
    scales = torch.empty((batch, heads, rows, 8), device=k.device, dtype=torch.float8_e4m3fn)
    _smooth_pack[(triton.cdiv(rows, 4), batch, heads)](
        k, k, mean, k, codes, scales, k, k, *k.stride()[:3], *k.stride()[:3],
        batch, rows, heads, 0, 4, PREPARE_PREFIX=False, num_warps=4,
    )
    return codes, scales
