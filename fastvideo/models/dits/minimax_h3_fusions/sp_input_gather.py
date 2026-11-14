# SPDX-License-Identifier: Apache-2.0
"""Gather exact packed inputs before head-sharded two-rank H3 projections.

The packed inputs already carry the shared unit activation scale. Only weight
views are sharded; each row/head computes the same Q/K/V projection, norm/RoPE,
and transport FP8 rounding as the sequence-sharded producer. Scale reassembly
handles the unaligned join between independently padded 128-row scale tiles.
"""
import torch
import triton
import triton.language as tl

from fastvideo.distributed.async_collectives import all_gather_tensor_async
from fastvideo.distributed.projection_exchange import completed_projection_packets
from fastvideo.layers.quantization.nvfp4_config import _mm_fp4, _mm_fp4_backend

from .sp_projection import allocate
from .sp_qkv_ordered import _normalize


@triton.jit
def _join_scale_rows(Source, Out, ROWS: tl.constexpr, PADDED: tl.constexpr, GROUPS: tl.constexpr,
                     TOTAL_PADDED: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, group = index // GROUPS, index % GROUPS
    rank, local_row = row // ROWS, row % ROWS
    local_tile = (local_row // 128) * (GROUPS // 4) + group // 4
    local_index = ((local_tile * 32 + local_row % 32) * 4 + (local_row // 32) % 4) * 4 + group % 4
    value = tl.load(Source + rank * PADDED * GROUPS + local_index, row < 2 * ROWS, other=0)
    tile = (row // 128) * (GROUPS // 4) + group // 4
    destination = ((tile * 32 + row % 32) * 4 + (row // 32) % 4) * 4 + group % 4
    tl.store(Out + destination, value, index < TOTAL_PADDED * GROUPS)


def join_scale_rows(source: torch.Tensor, rows: int) -> torch.Tensor:
    assert source.ndim == 3 and source.shape[0] == 2 and source.is_contiguous()
    assert source.dtype == torch.uint8 and source.shape[2] % 4 == 0
    padded, groups = source.shape[1:]
    assert padded == triton.cdiv(rows, 128) * 128
    total_padded = triton.cdiv(rows * 2, 128) * 128
    output = torch.empty((total_padded, groups), device=source.device, dtype=source.dtype)
    _join_scale_rows[(triton.cdiv(total_padded * groups, 512), )](
        source, output, rows, padded, groups, total_padded, 512, num_warps=4)
    return output


@triton.jit
def _pack_gathered(X, Weight, Cos, Sin, Payload, Scale, LOCAL_ROWS: tl.constexpr, HEADS: tl.constexpr,
                   CHUNKS: tl.constexpr, ROTARY: tl.constexpr, EPS: tl.constexpr, PROJECTION: tl.constexpr,
                   BLOCK_ROWS: tl.constexpr):
    row_head = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    row, head = row_head // HEADS, row_head % HEADS
    column = tl.arange(0, 128)
    valid = row[:, None] < LOCAL_ROWS * 2
    if PROJECTION == 2:
        values = tl.load(X + row_head[:, None] * 128 + column[None, :], valid, other=0.0).to(tl.float32)
    else:
        values = _normalize(X, Weight, Cos, Sin, row, head, HEADS, LOCAL_ROWS * 2, 128, ROTARY, EPS, 128)
    maximum = tl.max(tl.abs(values), 1)
    scale = tl.maximum(maximum * (1.0 / 448.0), 1e-12)
    quantized = (values / scale[:, None]).to(tl.float8e4nv)
    chunk_heads: tl.constexpr = HEADS // CHUNKS
    chunk = head // chunk_heads
    source = row // LOCAL_ROWS
    destination = ((
        (chunk * 2 + source) * 3 + PROJECTION) * LOCAL_ROWS + row % LOCAL_ROWS) * chunk_heads + head % chunk_heads
    tl.store(Payload + destination[:, None] * 128 + column[None, :], quantized, valid)
    tl.store(Scale + destination, scale, row < LOCAL_ROWS * 2)


def pack_gathered(x, payload, scale, projection, weight=None, cosine=None, sine=None, epsilon=0.0) -> None:
    chunks, world, projections, rows, heads, dim = payload.shape
    assert world == 2 and projections == 3 and dim == 128
    assert x.shape == (2 * rows, chunks * heads * dim) and x.is_contiguous()
    assert payload.is_contiguous() and scale.is_contiguous()
    if projection == 2:
        weight, cosine, sine = x, x, x
        rotary = 0
    else:
        assert weight.shape == (128, ) and cosine.shape == sine.shape
        assert cosine.shape[0] == rows * 2 and cosine.is_contiguous() and sine.is_contiguous()
        rotary = cosine.shape[1]
    block_rows = 16 if projection == 2 else 8
    _pack_gathered[(triton.cdiv(2 * rows * chunks * heads, block_rows), )](
        x, weight, cosine, sine, payload, scale, rows, chunks * heads, chunks,
        rotary, epsilon, projection, block_rows, num_warps=4 if projection == 2 else 8)


def supports_input_gather(attn, hidden: torch.Tensor) -> bool:
    """Keep other shapes/backends on the established projection exchange."""
    if (not hidden.is_cuda or hidden.dtype != torch.bfloat16
            or hidden.shape != (1, 7430, 5376) or attn.num_attention_heads != 56
            or torch.cuda.get_device_capability(hidden.device) != (12, 1)
            or _mm_fp4_backend() != "cudnn"):
        return False
    for linear in (attn.to_q, attn.to_k, attn.to_v):
        weight = getattr(linear, "_nvfp4_weight", None)
        scale = getattr(linear, "_nvfp4_weight_scale", None)
        if (weight is None or scale is None or weight.shape != (7168, 2688)
                or scale.shape != (7168, 336) or linear.bias is not None
                or weight.dtype != torch.uint8 or scale.dtype != torch.uint8
                or weight.device != hidden.device or scale.device != hidden.device
                or not weight.is_contiguous() or not scale.is_contiguous()):
            return False
    return True


def project_gathered_inputs(attn, hidden, rotary, sp_group, chunks, pre_quantized=None):
    """Return completed head packets and the gate computed while inputs travel.

    The caller checks eager, two-rank, head-128 inference and the common unit
    activation-scale contract through _shared_qkv_gate_scale. No activations
    or packets survive this call except through the returned consumer handles.
    """
    linears = (attn.to_q, attn.to_k, attn.to_v)
    pre = pre_quantized if pre_quantized is not None else linears[0].quant_method.quantize_input(hidden)
    a, sa, input_scale = pre
    a, sa = a.reshape(-1, a.shape[-1]), sa.reshape(-1, sa.shape[-1])
    cosine, sine = (item.to(hidden.dtype) for item in rotary)
    pending = [all_gather_tensor_async(value, sp_group.device_group, 2) for value in (a, sa, cosine, sine)]
    linear = attn.to_gate_compress
    gate = linear.quant_method.apply(linear, hidden, linear.bias, pre_quantized=pre)
    gathered_a, gathered_sa, gathered_cosine, gathered_sine = (item.wait() for item in pending)
    full_a = gathered_a.flatten(0, 1)
    full_sa = join_scale_rows(gathered_sa, a.shape[0])
    full_cosine, full_sine = gathered_cosine.flatten(0, 1), gathered_sine.flatten(0, 1)
    rows, heads, rank = hidden.shape[1], attn.num_attention_heads, sp_group.rank_in_group
    payload, scale = allocate(rows, heads, 2, chunks, hidden.device)
    for projection, linear in enumerate(linears):
        half = linear._nvfp4_weight.shape[0] // 2
        weight = linear._nvfp4_weight.narrow(0, rank * half, half)
        weight_scale = linear._nvfp4_weight_scale.narrow(0, rank * half, half)
        alpha = (linear._nvfp4_alpha / input_scale if hasattr(linear, "_nvfp4_alpha")
                 else 1.0 / (input_scale * linear._weight_global_sf))
        value = _mm_fp4(full_a, weight.T, full_sa, weight_scale.T, alpha, torch.bfloat16, None, backend="cudnn")
        if projection == 2:
            pack_gathered(value, payload, scale, projection)
        else:
            norm = attn.norm_q if projection == 0 else attn.norm_k
            pack_gathered(value, payload, scale, projection, norm.weight, full_cosine, full_sine, norm.eps)
    return completed_projection_packets(payload, scale, rank), gate
