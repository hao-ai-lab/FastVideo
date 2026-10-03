# SPDX-License-Identifier: Apache-2.0
"""Inference fast path: VSA-H3 attention on the block-sparse FP4 kernel.

Opt-in with ``FASTVIDEO_H3_VSA_FP4=1`` (no-grad, single sequence-parallel
rank, ``fastvideo-kernel`` built with ``attn_qat_infer``). The selection is
VSA-H3's own: tile pooling, top-k block mask and the gated compression branch
are unchanged; only the block-sparse attention itself runs on SageAttention3's
FP4 kernel (BF16 Triton otherwise), with 64-token tiles carried by quadrant
masks on the kernel's 128x128 blocks.

The attention input is gathered into tile order once per block (one
``hidden_size``-wide pass; pad rows stay zero, so q/k/v pad rows are exactly
zero through the bias-free projections, RMSNorm and RoPE). That replaces the
generic path's concat, four tile scatters and three transposes, and lets q, k
and v share one activation quantization. The output returns to packed order
with one gather before ``to_out``.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import (MiniMaxH3VSAMetadata, _build_block_mask, _pool_tiles)

VSA_FP4_ENV = "FASTVIDEO_H3_VSA_FP4"
_BLOCK = 128


class _StageTimer:
    """Opt-in (``FASTVIDEO_H3_SP_PROFILE=1``) CUDA-event spans summed per stage over one DiT forward.

    Events are only recorded during the forward; ``flush`` synchronizes once
    and returns milliseconds per stage, so enabling it costs one sync per step.
    """

    def __init__(self) -> None:
        self.enabled = os.environ.get("FASTVIDEO_H3_SP_PROFILE", "0") == "1"
        self._spans: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def span(self, name: str):
        import contextlib
        if not self.enabled:
            return contextlib.nullcontext()
        return self._record(name)

    def _record(self, name: str):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self._spans.append((name, start, end))

        return ctx()

    def flush(self) -> dict[str, float]:
        torch.cuda.synchronize()
        totals: dict[str, float] = {}
        for name, start, end in self._spans:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end)
        self._spans.clear()
        return {k: round(v, 1) for k, v in totals.items()}


STAGES = _StageTimer()

_fp4_api: Any = None


def vsa_fp4_requested() -> bool:
    return os.environ.get(VSA_FP4_ENV, "0") == "1"


def _api() -> Any:
    global _fp4_api
    if _fp4_api is None:
        import attn_qat_infer.api as api
        _fp4_api = api
    return _fp4_api


class _TileLayout:
    """Per-step tile-order state shared by every block of one forward."""

    def __init__(self, meta: MiniMaxH3VSAMetadata, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.tile = int(meta.tile_elems)
        self.n_tiles = int(meta.variable_block_sizes.numel())
        self.rows = math.ceil(self.n_tiles * self.tile / _BLOCK) * _BLOCK
        self.untile = meta.untile_combined_index
        self.row_tile = self.untile // self.tile
        self.rotary_src = rotary_emb
        cos, sin = rotary_emb
        self.cos = cos.new_zeros((self.rows, cos.shape[-1])).index_copy_(0, self.untile, cos)
        self.sin = sin.new_zeros((self.rows, sin.shape[-1])).index_copy_(0, self.untile, sin)
        self._buf: torch.Tensor | None = None

    def gather_in(self, x: torch.Tensor) -> torch.Tensor:
        """Packed ``[B, L, C]`` -> tile-ordered ``[B, rows, C]``; pad rows stay zero.

        The buffer is reused across blocks: pad rows are never written and
        every valid row is overwritten, and each block consumes it (q/k/v
        projections) before the next block refills it.
        """
        shape = (x.shape[0], self.rows, x.shape[-1])
        if self._buf is None or self._buf.shape != shape or self._buf.dtype != x.dtype:
            self._buf = x.new_zeros(shape)
        return self._buf.index_copy_(1, self.untile, x)


def _layout_for(meta: MiniMaxH3VSAMetadata, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> _TileLayout:
    layout = getattr(meta, "_h3_fp4_layout", None)
    if layout is None or layout.rotary_src[0] is not rotary_emb[0]:
        layout = _TileLayout(meta, rotary_emb)
        meta._h3_fp4_layout = layout  # type: ignore[attr-defined]
    return layout


def _shared_input_projections(linears: tuple[Any, ...], x: torch.Tensor) -> list[torch.Tensor]:
    """Run projections of one input, quantizing it once when all are NVFP4 with the unit activation scale.

    Only then is one quantized copy exactly what each layer would have produced; layers with a calibrated
    or dynamic activation scale quantize their own input.
    """
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

    methods = [linear.quant_method for linear in linears]
    if not all(
            type(m) is NVFP4QuantizeMethod and m.wants_prequantized_input() and m.uses_unit_activation_scale(linear)
            for m, linear in zip(methods, linears, strict=True)):
        return [linear(x)[0] for linear in linears]
    pre = methods[0].quantize_input(x)
    return [m.apply(linear, x, linear.bias, pre_quantized=pre) for m, linear in zip(methods, linears, strict=True)]


def vsa_fp4_attention(attn: Any, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor],
                      meta: MiniMaxH3VSAMetadata, use_fused_rope: bool) -> torch.Tensor:
    """Attention core for ``MiniMaxH3Attention``; returns the pre-``to_out`` ``[B, L, H*D]``."""
    api = _api()
    layout = _layout_for(meta, rotary_emb)
    heads, dim = attn.num_attention_heads, attn.attention_head_dim
    with STAGES.span("qkv_proj_rope"):
        x_tiles = layout.gather_in(hidden_states)
        query, key, value = (t.unflatten(-1, (heads, dim))
                             for t in _shared_input_projections((attn.to_q, attn.to_k, attn.to_v), x_tiles))
        if use_fused_rope:
            from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
            cos, sin = layout.cos.to(query.dtype), layout.sin.to(query.dtype)
            query = fused_qknorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
            key = fused_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
        else:
            rope = (layout.cos, layout.sin)
            query = attn._apply_rotary_emb(attn.norm_q(query), rope)
            key = attn._apply_rotary_emb(attn.norm_k(key), rope)

    sim_fp8 = os.environ.get(SIM_SP_FP8_ENV, "0") == "1"
    if sim_fp8:
        query, key, value = (_fp8_roundtrip(t) for t in (query, key, value))

    vbs = meta.variable_block_sizes
    logical = layout.n_tiles * layout.tile
    with STAGES.span("select_mask"):
        q_pooled = _pool_tiles(query[:, :logical], vbs, layout.tile)
        k_pooled = _pool_tiles(key[:, :logical], vbs, layout.tile)
        scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (dim**0.5)
        sparsity = 0.0 if attn._layer_idx in meta.dense_layers else meta.VSA_sparsity
        mask = _build_block_mask(scores, meta.num_prefix_tiles, sparsity, meta.exempt, meta.video_tile_spans,
                                 meta.span_sparsities)
        q2k_idx, q2k_num, kv_valid, q2k_quad = api.vsa_tile_mask_to_fp4_blocks(mask, layout.tile, vbs)
    with STAGES.span("fp4_attention"):
        out = api.sageattn_blackwell_sparse_bshd(query, key, value, q2k_idx, q2k_num, kv_valid, q2k_quad)
    with STAGES.span("out_untile"):
        out = out.transpose(1, 2).index_select(1, layout.untile)  # [B, L, H, D], packed order
    if sim_fp8:
        out = _fp8_roundtrip(out)

    if attn.to_gate_compress is not None and attn._gate_active():
        with STAGES.span("gate_compress"):
            gate, _ = attn.to_gate_compress(hidden_states)
            v_pooled = _pool_tiles(value[:, :logical], vbs, layout.tile)
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled).permute(0, 2, 1, 3).to(out.dtype)
            out = out.addcmul_(out_c.index_select(1, layout.row_tile), gate.unflatten(-1, (heads, dim)))
    return out.flatten(2, 3)


# ---------------------------------------------------------------------------
# Ulysses sequence parallelism: FP8 head/sequence exchange around the FP4 core
# ---------------------------------------------------------------------------
#
# PCIe-only boxes (e.g. 8x RTX PRO 6000) move ~21 GB/s per GPU in an
# all-to-all, so the Ulysses exchange, not compute, bounds multi-GPU latency.
# q/k/v travel as FP8 (one scale per token and head): the attention kernel
# re-quantizes them to FP4 on arrival, so FP8 transport adds error far below
# that floor while halving the bytes. The VSA gate never travels: each rank
# applies it to its own sequence rows after an all-gather of the tiny per-tile
# compression output. The attention output returns as FP8 too; ``to_out``
# re-quantizes it to FP4.

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0

# Debug: apply the SP path's FP8 rounding on one GPU (q/k/v after RoPE and the
# attention output, one scale per token and head), to separate exchange
# rounding from sharding errors when comparing SP>1 against SP=1.
SIM_SP_FP8_ENV = "FASTVIDEO_H3_SIM_SP_FP8"


@torch.compile(dynamic=True, fullgraph=True)
def _fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """``[..., H, D]`` BF16 -> FP8 with a per-(token, head) scale -> BF16, as the SP exchange does."""
    xf = x.float()
    scale = (xf.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-12)
    return ((xf / scale[..., None]).to(_FP8).float() * scale[..., None]).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _pack_heads_fp8(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    world: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, H, D]`` x3 -> per-destination ``[W, 3, rows, H/W, D]`` FP8 payload and scales."""
    x = torch.stack((query, key, value)).float()
    scale = (x.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-12)
    payload = (x / scale[..., None]).to(_FP8)
    _, rows, heads, dim = x.shape
    payload = payload.view(3, rows, world, heads // world, dim).permute(2, 0, 1, 3, 4).contiguous()
    scale = scale.view(3, rows, world, heads // world).permute(2, 0, 1, 3).contiguous()
    return payload, scale


@torch.compile(dynamic=False, fullgraph=True)
def _unpack_seq_fp8(payload: torch.Tensor, scale: torch.Tensor, seq_len: int) -> torch.Tensor:
    """``[W, 3, rows, Hs, D]`` from every source rank -> ``[3, seq_len, Hs, D]`` BF16 in packed order."""
    x = payload.float() * scale[..., None]
    world, _, rows, heads, dim = x.shape
    return x.permute(1, 0, 2, 3, 4).reshape(3, world * rows, heads, dim)[:, :seq_len].to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _pack_seq_fp8(out_bhsd: torch.Tensor, untile: torch.Tensor, world: int,
                  rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile-ordered ``[1, Hs, R, D]`` -> packed, padded, per-destination ``[W, rows, Hs, D]`` FP8 + scales."""
    x = out_bhsd[0].transpose(0, 1).index_select(0, untile).float()  # [L, Hs, D]
    x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, world * rows - x.shape[0]))
    scale = (x.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-12)
    payload = (x / scale[..., None]).to(_FP8)
    return payload.view(world, rows, *payload.shape[1:]), scale.view(world, rows, scale.shape[-1])


@torch.compile(dynamic=False, fullgraph=True)
def _unpack_heads_fp8(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``[W, rows, Hs, D]`` from every head group -> ``[rows, W*Hs, D]`` BF16."""
    x = payload.float() * scale[..., None]
    world, rows, heads, dim = x.shape
    return x.permute(1, 0, 2, 3).reshape(rows, world * heads, dim).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _apply_gate(out: torch.Tensor, out_c: torch.Tensor, row_tile: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``out + out_c[row_tile] * gate`` for one rank's rows; out/gate ``[rows, H, D]``, out_c ``[n_tiles, H, D]``."""
    return out + out_c.index_select(0, row_tile) * gate


def _all_to_all(payload: torch.Tensor, scale: torch.Tensor, group: Any) -> tuple[torch.Tensor, torch.Tensor]:
    import torch.distributed as dist
    recv = torch.empty_like(payload)
    recv_scale = torch.empty_like(scale)
    dist.all_to_all_single(recv.view(torch.uint8), payload.view(torch.uint8), group=group)
    dist.all_to_all_single(recv_scale, scale, group=group)
    return recv, recv_scale


class _SPTileLayout:
    """Per-step tile-order state for one rank's head subset."""

    def __init__(self, meta: MiniMaxH3VSAMetadata, rank: int, local_rows: int) -> None:
        self.tile = int(meta.tile_elems)
        self.n_tiles = int(meta.variable_block_sizes.numel())
        self.rows = math.ceil(self.n_tiles * self.tile / _BLOCK) * _BLOCK
        self.seq_len = int(meta.total_seq_length)
        self.untile = meta.untile_combined_index
        row_tile = self.untile // self.tile
        local = torch.arange(rank * local_rows, (rank + 1) * local_rows, device=row_tile.device)
        # Rows past the sequence are SP padding; their outputs are discarded.
        self.local_row_tile = row_tile[local.clamp_max(self.seq_len - 1)]
        self._buf: torch.Tensor | None = None

    def tiles_from(self, qkv: torch.Tensor) -> torch.Tensor:
        """``[3, L, Hs, D]`` packed -> ``[3, rows, Hs, D]`` tile order with zero padding (reused buffer)."""
        shape = (3, self.rows, *qkv.shape[2:])
        if self._buf is None or self._buf.shape != shape:
            self._buf = qkv.new_zeros(shape)
        return self._buf.index_copy_(1, self.untile, qkv)


def vsa_fp4_attention_sp(attn: Any, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor],
                         meta: MiniMaxH3VSAMetadata, use_fused_rope: bool, sp_group: Any) -> torch.Tensor:
    """Ulysses-SP attention core on local sequence rows ``[1, rows, C]``; returns pre-``to_out`` ``[1, rows, H*D]``."""
    import torch.distributed as dist

    api = _api()
    world, rank = sp_group.world_size, sp_group.rank_in_group
    heads, dim = attn.num_attention_heads, attn.attention_head_dim
    local_rows = hidden_states.shape[1]
    layout = getattr(meta, "_h3_fp4_sp_layout", None)
    if layout is None:
        layout = _SPTileLayout(meta, rank, local_rows)
        meta._h3_fp4_sp_layout = layout  # type: ignore[attr-defined]

    with STAGES.span("qkv_proj_rope"):
        query, key, value = (t.unflatten(-1, (heads, dim))
                             for t in _shared_input_projections((attn.to_q, attn.to_k, attn.to_v), hidden_states))
        if use_fused_rope:
            from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
            cos, sin = rotary_emb[0].to(query.dtype), rotary_emb[1].to(query.dtype)
            query = fused_qknorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
            key = fused_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
        else:
            query = attn._apply_rotary_emb(attn.norm_q(query), rotary_emb)
            key = attn._apply_rotary_emb(attn.norm_k(key), rotary_emb)

    with STAGES.span("qkv_pack"):
        payload, scale = _pack_heads_fp8(query[0], key[0], value[0], world)
    with STAGES.span("qkv_all_to_all"):
        payload, scale = _all_to_all(payload, scale, sp_group.device_group)
    with STAGES.span("qkv_unpack_tile"):
        qkv = layout.tiles_from(_unpack_seq_fp8(payload, scale, layout.seq_len))  # [3, R, Hs, D]
    q_t, k_t, v_t = qkv[0:1], qkv[1:2], qkv[2:3]

    vbs = meta.variable_block_sizes
    logical = layout.n_tiles * layout.tile
    with STAGES.span("select_mask"):
        scores = torch.matmul(_pool_tiles(q_t[:, :logical], vbs, layout.tile),
                              _pool_tiles(k_t[:, :logical], vbs, layout.tile).transpose(-2, -1)) / (dim**0.5)
        sparsity = 0.0 if attn._layer_idx in meta.dense_layers else meta.VSA_sparsity
        mask = _build_block_mask(scores, meta.num_prefix_tiles, sparsity, meta.exempt, meta.video_tile_spans,
                                 meta.span_sparsities)
        q2k_idx, q2k_num, kv_valid, q2k_quad = api.vsa_tile_mask_to_fp4_blocks(mask, layout.tile, vbs)
    with STAGES.span("fp4_attention"):
        out_bhsd = api.sageattn_blackwell_sparse_bshd(q_t, k_t, v_t, q2k_idx, q2k_num, kv_valid, q2k_quad)

    with STAGES.span("out_pack"):
        payload, scale = _pack_seq_fp8(out_bhsd, layout.untile, world, local_rows)
    with STAGES.span("out_all_to_all"):
        payload, scale = _all_to_all(payload, scale, sp_group.device_group)
    with STAGES.span("out_unpack"):
        out = _unpack_heads_fp8(payload, scale)  # [rows, H, D]

    if attn.to_gate_compress is not None and attn._gate_active():
        with STAGES.span("gate_compress"):
            v_pooled = _pool_tiles(v_t[:, :logical], vbs, layout.tile)
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)[0].to(out.dtype)  # [Hs, n_tiles, D]
            gathered = torch.empty((world, *out_c.shape), dtype=out_c.dtype, device=out_c.device)
        with STAGES.span("gate_all_gather"):
            dist.all_gather_into_tensor(gathered, out_c.contiguous(), group=sp_group.device_group)
        with STAGES.span("gate_apply"):
            out_c_all = gathered.flatten(0, 1).transpose(0, 1)  # [n_tiles, H, D]
            gate, _ = attn.to_gate_compress(hidden_states)
            out = _apply_gate(out, out_c_all, layout.local_row_tile, gate[0].unflatten(-1, (heads, dim)))
    return out.flatten(1, 2).unsqueeze(0)
