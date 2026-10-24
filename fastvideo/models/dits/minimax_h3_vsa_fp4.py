# SPDX-License-Identifier: Apache-2.0
"""Inference fast path: VSA-H3 attention on the block-sparse FP4 kernel.

Opt-in with ``h3_vsa_fp4.enabled`` or ``FASTVIDEO_H3_VSA_FP4=1`` (no-grad,
eager inference, ``fastvideo-kernel`` built with ``attn_qat_infer``). The selection is
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
from collections.abc import Callable
from typing import Any

import torch

import fastvideo.envs as envs

from fastvideo.attention.backends.video_sparse_attn_h3 import (MiniMaxH3VSAMetadata, _build_block_mask, _pool_tiles)


def _tile_mask_to_fp4_blocks(api: Any, mask: torch.Tensor, tile: int, valid: torch.Tensor):
    if (mask.is_cuda and mask.dtype == torch.bool and mask.shape[-1] == mask.shape[-2]
            and tile > 0 and tile % 64 == 0 and valid.is_contiguous()
            and not torch.compiler.is_compiling()):
        from fastvideo.models.dits.minimax_h3_fusions.block_lists import tile_mask_to_fp4_blocks

        return tile_mask_to_fp4_blocks(mask, tile, valid)
    return api.vsa_tile_mask_to_fp4_blocks(mask, tile, valid)


def _fp4_kernel_kwargs(two_level_p: bool | None = None) -> dict[str, object]:
    """Quantization options for the FP4 video kernel.

    The native route defaults to single-level P; the QAD student trained with
    two-level P, selected here by an inference-only switch.
    """
    kwargs: dict[str, object] = {}
    enabled = envs.FASTVIDEO_H3_VSA_FP4_TWO_LEVEL_P.get() if two_level_p is None else two_level_p
    if enabled:
        kwargs["single_level_p_quant"] = False
    return kwargs


def _sparse_fp4_attention(api: Any, *args: Any, two_level_p: bool | None = None,
                          smooth_k: bool | None = None,
                          prepared_key: tuple[torch.Tensor, torch.Tensor] | None = None,
                          prepared_query: tuple[torch.Tensor, torch.Tensor] | None = None,
                          prepared_value: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Run the block-sparse FP4 kernel on lists built by vsa_tile_mask_to_fp4_blocks.

    Those lists are in range by construction, so skip the host-sync validation where the installed
    fastvideo-kernel supports turning it off; older releases have no ``validate`` argument.

    ``FASTVIDEO_H3_VSA_FP4_K_SMOOTH`` subtracts the per-head key mean before the kernel
    quantizes K. The shift is exact under softmax, so it only changes quantization error.
    """
    smooth = envs.FASTVIDEO_H3_VSA_FP4_K_SMOOTH.get() if smooth_k is None else smooth_k
    key = args[1]
    if (smooth and prepared_key is None and key.is_cuda and key.dtype == torch.bfloat16
            and key.shape[-1] == 128 and key.shape[1] % 128 == 0 and key.stride(-1) == 1
            and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(key.device)[0] >= 10
            and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                    "scale_and_quant_fp4", "scale_and_quant_fp4_transpose"))):
        from fastvideo.models.dits.minimax_h3_fusions.key_quant import quantize_smoothed_keys_only

        prepared_key = quantize_smoothed_keys_only(key, key.mean(dim=1, keepdim=True))
        smooth = False
    if smooth:
        args = (args[0], key - key.mean(dim=1, keepdim=True), *args[2:])
    if prepared_key is not None or prepared_query is not None or prepared_value is not None:
        query, key, value, indices, counts, valid, quadrants = args
        delta = api._zero_delta_s(query.shape[0], query.shape[2], key.shape[1], query.device)
        single_level = _fp4_kernel_kwargs(two_level_p).get("single_level_p_quant", True)
        return api.blockscaled_fp4_attn_sparse(
            prepared_query if prepared_query is not None else api.scale_and_quant_fp4(query.transpose(1, 2)),
            prepared_key if prepared_key is not None else api.scale_and_quant_fp4_permute(key.transpose(1, 2)),
            prepared_value if prepared_value is not None else api.scale_and_quant_fp4_transpose(value.transpose(1, 2)),
            delta, key.shape[1],
            indices, counts, valid, quadrants, False, query.dtype == torch.bfloat16, single_level,
        )[0]
    try:
        return api.sageattn_blackwell_sparse_bshd(*args, validate=False, **_fp4_kernel_kwargs(two_level_p))
    except TypeError as error:
        if "validate" not in str(error):
            raise
        return api.sageattn_blackwell_sparse_bshd(*args, **_fp4_kernel_kwargs(two_level_p))

_BLOCK = 128


def _resolve_prefix_mode(mode: str, steps: int, step: int) -> str:
    """The prefix split for one denoising step; ``steps < 0`` applies it to every step."""
    return mode if steps < 0 or step < steps else ""


class _StageTimer:
    """Opt-in (``FASTVIDEO_H3_SP_PROFILE=1``) CUDA-event spans summed per stage over one DiT forward.

    Events are only recorded during the forward; ``flush`` synchronizes once
    and returns milliseconds per stage, so enabling it costs one sync per step.
    """

    def __init__(self) -> None:
        self.enabled = envs.FASTVIDEO_H3_SP_PROFILE.get()
        self.nvtx = envs.FASTVIDEO_NVTX_PROFILE.get() and torch.cuda.is_available()
        self._spans: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def span(self, name: str):
        import contextlib
        if self.enabled:
            return self._record(name)
        if self.nvtx:
            from fastvideo.profiler import nvtx_range
            return nvtx_range(f"h3.{name}")
        return contextlib.nullcontext()

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
    return envs.FASTVIDEO_H3_VSA_FP4.get()


def _api() -> Any:
    global _fp4_api
    if _fp4_api is None:
        try:
            import attn_qat_infer.api as api
        except (ImportError, OSError) as error:
            raise RuntimeError("MiniMax-H3 FP4 attention was requested but attn_qat_infer could not load. "
                               "Build fastvideo-kernel for this GPU from the same checkout; "
                               "disable FP4 attention explicitly to use BF16 VSA.") from error
        _fp4_api = api
    return _fp4_api


class _KeyIndexCache:
    """The prefix split's valid-key positions, computed once per step layout.

    ``kv_valid`` comes from the tile sizes alone, so every block of the forward
    gets the same index; only the first block pays the host sync.
    """
    tile: int
    n_tiles: int
    _valid_key_index: torch.Tensor | None = None
    _valid_key_inverse: torch.Tensor | None = None

    def valid_key_index(self, kv_valid: torch.Tensor, half: int) -> torch.Tensor:
        if self._valid_key_index is None:
            self._valid_key_index = _valid_key_index(kv_valid, half, self.n_tiles * self.tile)
        return self._valid_key_index

    def valid_keys_from_tiles(self, tile_valid: torch.Tensor, half: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare prefix maps before mask selection, splitting 128-token tile counts into native halves."""
        native_valid = tile_valid
        if self._valid_key_index is None and self.tile != half:
            offsets = torch.arange(self.tile // half, device=tile_valid.device) * half
            native_valid = (tile_valid[:, None] - offsets[None, :]).clamp(0, half).flatten()
        index = self.valid_key_index(native_valid, half)
        return index, self.valid_key_inverse(native_valid, half)

    def valid_key_inverse(self, kv_valid: torch.Tensor, half: int) -> torch.Tensor:
        if self._valid_key_inverse is None:
            index = self.valid_key_index(kv_valid, half)
            self._valid_key_inverse = torch.full((self.n_tiles * self.tile,), -1,
                                                 device=index.device, dtype=torch.int64)
            self._valid_key_inverse[index] = torch.arange(index.numel(), device=index.device)
        return self._valid_key_inverse


class _TileLayout(_KeyIndexCache):
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
        self._quant_inverse: torch.Tensor | None = None

    def gather_quantized(self, quantized: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reorder per-row FP4 codes and scales without expanding them back to BF16."""
        from fastvideo.models.dits.minimax_h3_fusions.quant_tile import tile_quantized

        return tile_quantized(quantized, scales, self.packed_row_inverse())

    def packed_row_inverse(self) -> torch.Tensor:
        """Geometry-only map from padded tile rows to compact projection rows."""
        if self._quant_inverse is None:
            self._quant_inverse = torch.full((self.rows,), -1, device=self.untile.device, dtype=torch.int64)
            self._quant_inverse[self.untile] = torch.arange(self.untile.numel(), device=self.untile.device)
        return self._quant_inverse

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
    """Share compatible FP8 preparation or unit-scale NVFP4 activation quantization.

    Calibrated or dynamic NVFP4 activation scales retain independent preparation.
    """
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod
    from fastvideo.layers.quantization.fp8_config import FP8QuantizeMethod

    methods = [getattr(linear, "quant_method", None) for linear in linears]
    if any(m is None for m in methods):  # LoRA wrappers keep quant_method on base_layer.
        return [linear(x)[0] for linear in linears]
    same_nvfp4 = all(type(m) is NVFP4QuantizeMethod and m.wants_prequantized_input()
                     and m.uses_unit_activation_scale(linear)
                     for m, linear in zip(methods, linears, strict=True))
    same_fp8 = all(type(m) is FP8QuantizeMethod and m.granularity == methods[0].granularity for m in methods)
    if not (same_nvfp4 or same_fp8):
        return [linear(x)[0] for linear in linears]
    pre = methods[0].quantize_input(x)
    return [m.apply(linear, x, linear.bias, pre_quantized=pre) for m, linear in zip(methods, linears, strict=True)]


def _shared_qkv_gate_scale(attn: Any, x: torch.Tensor) -> torch.Tensor | None:
    """Return the common unit scale when QKV and the gate can share packed input."""
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

    linears = (attn.to_q, attn.to_k, attn.to_v, attn.to_gate_compress)
    if attn.to_gate_compress is None or x.shape[0] != 1 or x.shape[-1] % 64 or not attn._gate_active():
        return None
    if not all(type(getattr(linear, "quant_method", None)) is NVFP4QuantizeMethod
               and linear.quant_method.wants_prequantized_input()
               and linear.quant_method.uses_unit_activation_scale(linear) for linear in linears):
        return None
    return linears[0].quant_method.x_global_sf


def _shared_qkv_gate_projections(
    attn: Any, x: torch.Tensor, layout: _TileLayout | None = None,
    pre_quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    """Share packed activation preparation across QKV and the compression gate.

    FP4 groups never cross rows, so tile reordering preserves every code and
    scale. SP projects local packed rows directly and omits the tile layout.
    Compatible fused modulation producers can supply the packed input.
    """
    if _shared_qkv_gate_scale(attn, x) is None:
        if pre_quantized is not None:
            raise RuntimeError("Prequantized H3 input requires compatible QKV and gate projections")
        return None
    linears = (attn.to_q, attn.to_k, attn.to_v, attn.to_gate_compress)
    pre = linears[0].quant_method.quantize_input(x) if pre_quantized is None else pre_quantized
    qkv_pre, shape_view = pre, x
    if layout is not None:
        q, scales = layout.gather_quantized(pre[0], pre[1])
        qkv_pre = (q, scales, pre[2])
        # apply(pre_quantized=...) only reads x.shape. This view supplies the
        # tile-order shape without allocating or scattering a full BF16 tensor.
        shape_view = x[:, :1].expand(-1, layout.rows, -1)
    projected = [linear.quant_method.apply(linear, shape_view, linear.bias, pre_quantized=qkv_pre)
                 for linear in linears[:3]]
    return projected, pre


@torch.compile(dynamic=True, fullgraph=True)
def _fp8_per_token(x: torch.Tensor) -> torch.Tensor:
    """``[..., D]`` BF16 -> per-(row, head) FP8 E4M3 -> BF16, the prefix Q/K precision of the QK study.

    Element for element the emulator's ``fp8_per_token_fake_quant``. The zero
    scale is spelled as a reciprocal rather than a clamp so that a row with a
    subnormal amax rounds the same way in both.
    """
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True) / _FP8_MAX
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    return (xf * inv).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).float().mul(scale).to(torch.bfloat16)


def _half_row_valid(kv_valid: torch.Tensor, half: int) -> torch.Tensor:
    """Per-half valid-key counts -> a ``[rows]`` bool over the tile-ordered, zero-padded keys."""
    return (torch.arange(half, device=kv_valid.device).unsqueeze(0) < kv_valid.unsqueeze(1)).reshape(-1)


def _valid_key_index(kv_valid: torch.Tensor, half: int, logical: int) -> torch.Tensor:
    """Positions of the valid keys among the first ``logical`` tile-ordered rows (a host sync)."""
    return _half_row_valid(kv_valid, half)[:logical].nonzero().squeeze(1)


def _prefix_dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, prefix_rows: int,
                            logical: int, valid_index: torch.Tensor, precision: str,
                            key_is_smoothed: bool = False,
                            prepared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                            query_is_fp8: bool = False) -> torch.Tensor:
    """Dense attention for the prefix query rows, ``[B, H, prefix_rows, D]``.

    VSA gives every prefix (non-video) query tile every key column, so those rows need no
    block lists. Running them off the FP4 kernel lets their Q/K stay at 8-bit precision,
    which is where most of the FP4 attention error sits (the dense prefix rows carry the
    text, reference and audio queries).

    K smoothing is exact - it shifts every score row by the same constant - and runs
    before the FP8 rounding, in the order the emulator and the QAD student use.
    """
    import torch.nn.functional as F

    q, k, v = query[:, :prefix_rows], key[:, :logical], value[:, :logical]
    if not key_is_smoothed:
        k = k - k.mean(dim=1, keepdim=True)
    # Attend only the valid keys. Gathering them keeps SDPA on its unmasked (flash) kernel; a boolean
    # mask over all rows falls back to the memory-efficient kernel, 3-4x slower at OmniRef prefix sizes.
    if prepared_kv is not None:
        q = _fp8_per_token(q) if precision == "fp8" and not query_is_fp8 else q
        k, v = prepared_kv
    elif (precision == "fp8" and k.is_cuda and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and k.dtype == torch.bfloat16 and v.dtype == k.dtype
            and k.shape[-1] == 128 and k.stride(-1) == 1 and v.stride(-1) == 1
            and torch.cuda.get_device_capability(k.device)[0] >= 10):
        from fastvideo.models.dits.minimax_h3_fusions.prefix_quant import quantize_gather_prefix_kv

        q = _fp8_per_token(q) if not query_is_fp8 else q
        k, v = quantize_gather_prefix_kv(k, v, valid_index)
    else:
        if precision == "fp8":
            q = _fp8_per_token(q) if not query_is_fp8 else q
            k = _fp8_per_token(k)
        k, v = k.index_select(1, valid_index), v.index_select(1, valid_index)
    if prefix_rows == 15360 and precision == "fp8":
        from fastvideo.models.dits.minimax_h3_fusions.prefix_attention import try_prefix_attention

        specialized = try_prefix_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        if specialized is not None:
            return specialized
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                          scale=1.0 / math.sqrt(q.shape[-1]))


def _fp4_fine_attention(api: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        q2k_idx: torch.Tensor, q2k_num: torch.Tensor, kv_valid: torch.Tensor,
                        q2k_quad: torch.Tensor | None, *, logical: int, prefix_rows: int, precision: str,
                        options: dict[str, Any], layout: _KeyIndexCache | None = None,
                        return_split: bool = False,
                        prepared_query: tuple[torch.Tensor, torch.Tensor] | None = None,
                        prepared_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                        prefix_query_is_fp8: bool = False,
                        value_index: torch.Tensor | None = None,
                        prepared_keys: tuple[
                            tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
                        ] | None = None,
                        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, int]:
    numerics = {"two_level_p": options.get("two_level_p"), "smooth_k": options.get("smooth_k")}
    if not prefix_rows:
        return _sparse_fp4_attention(api, query, key, value, q2k_idx, q2k_num, kv_valid, q2k_quad,
                                      prepared_query=prepared_query, prepared_value=prepared_value, **numerics)
    if not getattr(api, "KEY_LENGTH_SIZING", False):
        raise RuntimeError("the MiniMax-H3 FP4 prefix split needs a fastvideo-kernel build whose "
                           "sageattn_blackwell_sparse_bshd sizes by key length; rebuild fastvideo-kernel "
                           "from this checkout")
    if layout is not None:
        valid_index = layout.valid_key_index(kv_valid, api.HALF_N)
    else:
        valid_index = _valid_key_index(kv_valid, api.HALF_N, logical)
    smooth = numerics["smooth_k"]
    if smooth is None:
        smooth = envs.FASTVIDEO_H3_VSA_FP4_K_SMOOTH.get()
    shared_smoothing = bool(smooth and logical == key.shape[1])
    prepared_key, prepared_kv = None, None
    if prepared_keys is not None:
        prepared_key, prefix_key, prefix_value = prepared_keys
        prepared_kv = (prefix_key, prefix_value)
        numerics["smooth_k"] = False
    elif (shared_smoothing and precision == "fp8" and layout is not None and key.is_cuda
            and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and key.dtype == torch.bfloat16 and value.dtype == key.dtype and key.shape[-1] == 128
            and key.stride(-1) == 1 and value.stride(-1) == 1
            and torch.cuda.get_device_capability(key.device)[0] >= 10
            and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                    "scale_and_quant_fp4", "scale_and_quant_fp4_transpose"))):
        from fastvideo.models.dits.minimax_h3_fusions.key_quant import quantize_smoothed_keys

        inverse = layout.valid_key_inverse(kv_valid, api.HALF_N)
        prepared_key, prefix_key, prefix_value = quantize_smoothed_keys(
            key, value, key.mean(dim=1, keepdim=True), inverse, valid_index.numel(), value_index=value_index,
        )
        prepared_kv = (prefix_key, prefix_value)
        numerics["smooth_k"] = False
    elif shared_smoothing:
        # Both branches reduce the same rows with the same strides and dtype.
        # Keep one rounded BF16 subtraction result instead of repeating it.
        key = key - key.mean(dim=1, keepdim=True)
        numerics["smooth_k"] = False
    with STAGES.span("prefix_attention"):
        prefix_out = _prefix_dense_attention(query, key, value, prefix_rows, logical, valid_index, precision,
                                             key_is_smoothed=shared_smoothing, prepared_kv=prepared_kv,
                                             query_is_fp8=prefix_query_is_fp8)
    # A tile-64 prefix can end halfway through a native 128-row block. Keep that whole query block and its
    # original quadrant mask, then discard the overlapping prefix half and trailing padding from its output.
    head = prefix_rows // _BLOCK
    start = head * _BLOCK
    # With prepared FP4 queries, the raw suffix supplies only dtype/device/head
    # metadata; query may hold just the BF16 prefix without a video intermediate.
    video_out = _sparse_fp4_attention(api, query[:, start:], key, value,
                                      q2k_idx[:, :, head:].contiguous(), q2k_num[:, :, head:].contiguous(), kv_valid,
                                      None if q2k_quad is None else q2k_quad[:, :, head:].contiguous(),
                                      prepared_key=prepared_key, prepared_query=prepared_query,
                                      prepared_value=prepared_value, **numerics)
    if return_split:
        return prefix_out, video_out, start
    return torch.cat((prefix_out, video_out[:, :, prefix_rows - start:logical - start]), dim=2)


def _prefix_settings(attn: Any, meta: MiniMaxH3VSAMetadata, n_tiles: int) -> tuple[str, int]:
    options = getattr(attn, "_vsa_fp4_options", {})
    mode = options.get("prefix_precision")
    if mode is None:
        mode = envs.FASTVIDEO_H3_VSA_FP4_PREFIX.get()
    mode = _resolve_prefix_mode(mode, envs.FASTVIDEO_H3_VSA_FP4_PREFIX_STEPS.get(), int(meta.current_timestep))
    rows = (meta.num_prefix_tiles * int(meta.tile_elems)
            if mode and int(meta.tile_elems) in (64, 128) and 0 < meta.num_prefix_tiles < n_tiles else 0)
    return mode, rows


def vsa_tile_first_attention(attn: Any, hidden_states: torch.Tensor,
                             rotary_emb: tuple[torch.Tensor, torch.Tensor],
                             meta: MiniMaxH3VSAMetadata, use_fused_rope: bool) -> torch.Tensor:
    """Single-rank BF16 VSA with one input scatter instead of a Q/K/V/gate stack.

    The existing backend computes the same tile-64 mask, valid-key handling,
    fine attention and compression branch. Bias-free projections keep pad
    rows zero. This path is inference-only and keeps the checkpoint layout.
    """
    layout = _layout_for(meta, rotary_emb)
    logical = layout.n_tiles * layout.tile
    heads, dim = attn.num_attention_heads, attn.attention_head_dim
    with STAGES.span("tile_input"):
        x_tiles = layout.gather_in(hidden_states)[:, :logical]
    with STAGES.span("qkv_proj"):
        query, key, value = (t.unflatten(-1, (heads, dim))
                             for t in _shared_input_projections((attn.to_q, attn.to_k, attn.to_v), x_tiles))
    with STAGES.span("qknorm_rope"):
        cos, sin = layout.cos[:logical], layout.sin[:logical]
        if use_fused_rope:
            from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
            query = fused_qknorm_rope(query, attn.norm_q.weight, cos.to(query.dtype), sin.to(query.dtype), attn.norm_q.eps)
            key = fused_qknorm_rope(key, attn.norm_k.weight, cos.to(key.dtype), sin.to(key.dtype), attn.norm_k.eps)
        else:
            query = attn._apply_rotary_emb(attn.norm_q(query), (cos, sin))
            key = attn._apply_rotary_emb(attn.norm_k(key), (cos, sin))
    gate = None
    if attn.to_gate_compress is not None and attn._gate_active():
        with STAGES.span("gate_proj"):
            gate, _ = attn.to_gate_compress(x_tiles)
            gate = gate.unflatten(-1, (heads, dim))
    capture_root = envs.FASTVIDEO_H3_CAPTURE_QKV.get()
    if capture_root and attn._layer_idx in (0, 20, 41):
        from pathlib import Path
        root = Path(capture_root)
        root.mkdir(parents=True, exist_ok=True)
        capture = root / f"layer-{attn._layer_idx}.pt"
        if not capture.exists():
            q_pooled = _pool_tiles(query, meta.variable_block_sizes, meta.tile_elems)
            k_pooled = _pool_tiles(key, meta.variable_block_sizes, meta.tile_elems)
            scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / dim**0.5
            sparsity = 0.0 if attn._layer_idx in meta.dense_layers else meta.VSA_sparsity
            mask = _build_block_mask(scores, meta.num_prefix_tiles, sparsity, meta.exempt,
                                     meta.video_tile_spans, meta.span_sparsities)
            # Two heads keep the artifact small while retaining all real keys,
            # query rows, per-tile selections and partial-tile validity.
            torch.save({"q": query[:, :, :2].transpose(1, 2).contiguous().cpu(),
                        "k": key[:, :, :2].transpose(1, 2).contiguous().cpu(),
                        "v": value[:, :, :2].transpose(1, 2).contiguous().cpu(),
                        "mask": mask[:, :2].cpu(), "vbs": meta.variable_block_sizes.cpu(),
                        "untile": meta.untile_combined_index.cpu()}, capture)
            del q_pooled, k_pooled, scores, mask
    with STAGES.span("attention"):
        out = attn.distributed_attention.attn_impl.forward(query, key, value, gate, meta)
    with STAGES.span("untile_output"):
        return out.index_select(1, layout.untile).flatten(2, 3)


def vsa_fp4_attention(attn: Any, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor],
                      meta: MiniMaxH3VSAMetadata, use_fused_rope: bool,
                      pre_quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                      ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Attention core for ``MiniMaxH3Attention``; returns the pre-``to_out`` ``[B, L, H*D]``."""
    api = _api()
    layout = _layout_for(meta, rotary_emb)
    heads, dim = attn.num_attention_heads, attn.attention_head_dim
    prefix_mode, prefix_rows = _prefix_settings(attn, meta, layout.n_tiles)
    sim_fp8 = envs.FASTVIDEO_H3_SIM_SP_FP8.get()
    prepared_query, q_pooled = None, None
    prepared_key_mean = None
    prefix_query_is_fp8 = False
    smooth = getattr(attn, "_vsa_fp4_options", {}).get("smooth_k")
    smooth = envs.FASTVIDEO_H3_VSA_FP4_K_SMOOTH.get() if smooth is None else smooth
    # The measured cuDNN/GB10 route preserves every projected row while omitting
    # zero padding from QKV GEMMs. Other backends keep the tiled projection path.
    compact = (use_fused_rope and not sim_fp8 and layout.tile == 64
               and layout.rows == layout.n_tiles * layout.tile and dim == 128
               and hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
               and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
               and (not prefix_rows or (prefix_mode == "fp8" and smooth))
               and envs.FASTVIDEO_NVFP4_MM_BACKEND.get() == "cudnn"
               and torch.cuda.get_device_capability(hidden_states.device) == (12, 1)
               and _shared_qkv_gate_scale(attn, hidden_states) is not None
               and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                      "scale_and_quant_fp4", "scale_and_quant_fp4_permute",
                                                      "scale_and_quant_fp4_transpose")))
    input_index = layout.packed_row_inverse() if compact else None
    with STAGES.span("qkv_proj_rope"):
        shared = _shared_qkv_gate_projections(attn, hidden_states, None if compact else layout, pre_quantized)
        if shared is None:
            x_tiles = layout.gather_in(hidden_states)
            projected = _shared_input_projections((attn.to_q, attn.to_k, attn.to_v), x_tiles)
            gate_pre = None
        else:
            projected, gate_pre = shared
        query, key, value = (t.unflatten(-1, (heads, dim)) for t in projected)
        del projected, shared
        if compact:
            from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import gather_qknorm_rope

            cos, sin = layout.cos.to(query.dtype), layout.sin.to(query.dtype)
            if prefix_rows:
                from fastvideo.models.dits.minimax_h3_fusions.query_pool import prepare_pooled_queries

                query, prepared_query, q_pooled = prepare_pooled_queries(
                    query, attn.norm_q.weight, cos, sin, attn.norm_q.eps, input_index,
                    meta.variable_block_sizes, prefix_rows,
                )
            else:
                query, prepared_query = gather_qknorm_rope(
                    query, attn.norm_q.weight, cos, sin, attn.norm_q.eps, input_index, fp4_start=0,
                )
            if prefix_rows and key.shape[0] == 1 and heads == 56 and input_index.numel() == 18432 and cos.shape[1] == 96:
                from fastvideo.models.dits.minimax_h3_fusions.key_norm_mean import gather_key_norm_mean

                key, prepared_key_mean = gather_key_norm_mean(
                    key, attn.norm_k.weight, cos, sin, attn.norm_k.eps, input_index,
                )
            else:
                key, _ = gather_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps, input_index)
        elif use_fused_rope:
            from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
            cos, sin = layout.cos.to(query.dtype), layout.sin.to(query.dtype)
            if (not sim_fp8 and query.is_cuda and query.dtype == torch.bfloat16
                    and dim == 128 and query.shape[1] % 128 == 0
                    and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
                    and torch.cuda.get_device_capability(query.device)[0] >= 10
                    and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                            "scale_and_quant_fp4", "scale_and_quant_fp4_permute",
                                                            "scale_and_quant_fp4_transpose"))):
                from fastvideo.models.dits.minimax_h3_fusions.qknorm_rope import qknorm_rope_quantize_fp4

                if (layout.tile == 128 and query.shape[1] == layout.n_tiles * layout.tile
                        and prefix_mode == "fp8" and prefix_rows
                        and torch.cuda.get_device_capability(query.device) == (12, 1)):
                    from fastvideo.models.dits.minimax_h3_fusions.query_pool import prepare_pooled_queries

                    query, prepared_query, q_pooled = prepare_pooled_queries(
                        query, attn.norm_q.weight, cos, sin, attn.norm_q.eps, None,
                        meta.variable_block_sizes, prefix_rows, tile_elems=128, prefix_fp8=True, split_prefix=True,
                    )
                    prefix_query_is_fp8 = True
                else:
                    query, prepared_query = qknorm_rope_quantize_fp4(
                        query, attn.norm_q.weight, cos, sin, attn.norm_q.eps, prefix_rows // _BLOCK * _BLOCK,
                    )
            else:
                query = fused_qknorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
            key = fused_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
        else:
            rope = (layout.cos, layout.sin)
            query = attn._apply_rotary_emb(attn.norm_q(query), rope)
            key = attn._apply_rotary_emb(attn.norm_k(key), rope)

    if sim_fp8:
        query, key, value = (_fp8_roundtrip(t) for t in (query, key, value))

    vbs = meta.variable_block_sizes
    logical = layout.n_tiles * layout.tile
    prepared_keys, k_pooled = None, None
    v_pooled, prepared_value = None, None
    fused_gate = (attn.to_gate_compress is not None and attn._gate_active()
                  and not sim_fp8 and query.is_cuda and query.dtype == torch.bfloat16)
    omni_key_pool = (layout.tile == 128 and logical == key.shape[1] and prefix_mode == "fp8" and smooth
                     and not sim_fp8 and key.is_cuda and key.dtype == value.dtype == torch.bfloat16
                     and dim == 128 and key.stride(-1) == value.stride(-1) == 1
                     and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
                     and torch.cuda.get_device_capability(key.device)[0] >= 10
                     and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                             "scale_and_quant_fp4", "scale_and_quant_fp4_transpose")))
    if prefix_rows and (compact or omni_key_pool):
        from fastvideo.models.dits.minimax_h3_fusions.key_quant import pool_quantize_smoothed_keys

        valid_index, inverse = layout.valid_keys_from_tiles(vbs, api.HALF_N)
        key_mean = key.mean(dim=1, keepdim=True) if prepared_key_mean is None else prepared_key_mean
        if fused_gate and (compact or torch.cuda.get_device_capability(key.device) == (12, 1)):
            from fastvideo.models.dits.minimax_h3_fusions.kv_pool import pool_quantize_prefix_kv

            key_result, value_result = pool_quantize_prefix_kv(
                key, value, key_mean, inverse, valid_index.numel(), vbs, input_index,
                tile_elems=layout.tile,
            )
            packed_key, prefix_key, prefix_value, k_pooled = key_result
            v_pooled, prepared_value = value_result
        else:
            packed_key, prefix_key, prefix_value, k_pooled = pool_quantize_smoothed_keys(
                key, value, key_mean, inverse, valid_index.numel(), vbs,
                value_index=input_index, tile_elems=layout.tile,
            )
        prepared_keys = (packed_key, prefix_key, prefix_value)
    with STAGES.span("select_mask"):
        if q_pooled is None:
            q_pooled = _pool_tiles(query[:, :logical], vbs, layout.tile)
        if k_pooled is None:
            k_pooled = _pool_tiles(key[:, :logical], vbs, layout.tile)
        scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (dim**0.5)
        sparsity = 0.0 if attn._layer_idx in meta.dense_layers else meta.VSA_sparsity
        mask = _build_block_mask(scores, meta.num_prefix_tiles, sparsity, meta.exempt, meta.video_tile_spans,
                                 meta.span_sparsities)
        q2k_idx, q2k_num, kv_valid, q2k_quad = _tile_mask_to_fp4_blocks(api, mask, layout.tile, vbs)
    if (prepared_value is None and fused_gate and layout.tile in (64, 128) and (compact or logical == value.shape[1]) and dim == 128
            and value.stride(-1) == 1 and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(value.device)[0] >= 10
            and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                    "scale_and_quant_fp4", "scale_and_quant_fp4_permute"))):
        from fastvideo.models.dits.minimax_h3_fusions.value_quant import pool_quantize_values

        v_pooled, prepared_value = pool_quantize_values(value, vbs, input_index=input_index, tile_elems=layout.tile)
    with STAGES.span("fp4_attention"):
        out = _fp4_fine_attention(api, query, key, value, q2k_idx, q2k_num, kv_valid, q2k_quad,
                                  logical=logical, prefix_rows=prefix_rows, precision=prefix_mode,
                                  options=getattr(attn, "_vsa_fp4_options", {}), layout=layout,
                                  return_split=fused_gate, prepared_query=prepared_query, prepared_value=prepared_value,
                                  prefix_query_is_fp8=prefix_query_is_fp8,
                                  value_index=input_index, prepared_keys=prepared_keys)
    if not fused_gate:
        with STAGES.span("out_untile"):
            out = out.transpose(1, 2).index_select(1, layout.untile)  # [B, L, H, D], packed order
    if sim_fp8:
        out = _fp8_roundtrip(out)

    if attn.to_gate_compress is not None and attn._gate_active():
        with STAGES.span("gate_compress"):
            if gate_pre is None:
                gate, _ = attn.to_gate_compress(hidden_states)
            else:
                linear = attn.to_gate_compress
                gate = linear.quant_method.apply(linear, hidden_states, linear.bias, pre_quantized=gate_pre)
            if v_pooled is None:
                v_pooled = _pool_tiles(value[:, :logical], vbs, layout.tile)
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled).permute(0, 2, 1, 3).to(query.dtype)
            if fused_gate:
                from fastvideo.models.dits.minimax_h3_fusions.untile_gate import untile_gate
                from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

                method = getattr(attn.to_out, "quant_method", None)
                if (type(method) is NVFP4QuantizeMethod and method.wants_prequantized_input()
                        and method.uses_unit_activation_scale(attn.to_out) and gate.shape[-1] % 64 == 0):
                    prepared = untile_gate(out, out_c, gate, layout.untile, layout.row_tile, method.x_global_sf)
                    return gate, prepared
                return untile_gate(out, out_c, gate, layout.untile, layout.row_tile)
            gate = gate.unflatten(-1, (heads, dim))
            chunk = envs.FASTVIDEO_H3_GATE_CHUNK_ROWS.get()
            if chunk > 0 and out.shape[1] > chunk:
                # The full gather materializes [B, L, H, D] (1.2 GiB at 768p10) beside the 36L
                # decoder. Chunked gathers and the same elementwise FMA are bit-identical.
                for start in range(0, out.shape[1], chunk):
                    stop = min(start + chunk, out.shape[1])
                    out[:, start:stop].addcmul_(out_c.index_select(1, layout.row_tile[start:stop]),
                                                gate[:, start:stop])
            else:
                out = out.addcmul_(out_c.index_select(1, layout.row_tile), gate)
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


def _all_to_all(payload: torch.Tensor, scale: torch.Tensor, group: Any,
                overlap: Callable[[], None] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    import torch.distributed as dist
    recv = torch.empty_like(payload)
    recv_scale = torch.empty_like(scale)
    if overlap is None:
        dist.all_to_all_single(recv.view(torch.uint8), payload.view(torch.uint8), group=group)
        dist.all_to_all_single(recv_scale, scale, group=group)
    else:
        # Keep the current compute stream independent of NCCL until the local
        # projection has been enqueued. wait() then orders consumers of recv.
        payload_work = dist.all_to_all_single(recv.view(torch.uint8), payload.view(torch.uint8),
                                              group=group, async_op=True)
        scale_work = dist.all_to_all_single(recv_scale, scale, group=group, async_op=True)
        overlap()
        payload_work.wait()
        scale_work.wait()
    return recv, recv_scale


class _SPTileLayout(_KeyIndexCache):
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
        self._inverse: torch.Tensor | None = None

    def tiles_from(self, qkv: torch.Tensor) -> torch.Tensor:
        """``[3, L, Hs, D]`` packed -> ``[3, rows, Hs, D]`` tile order with zero padding (reused buffer)."""
        shape = (3, self.rows, *qkv.shape[2:])
        if self._buf is None or self._buf.shape != shape:
            self._buf = qkv.new_zeros(shape)
        return self._buf.index_copy_(1, self.untile, qkv)

    def fp8_buffers(self, payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the invariant row map and scratch tensor overwritten by each receive."""
        if self._inverse is None:
            self._inverse = torch.full((self.rows,), -1, device=payload.device, dtype=torch.int64)
            self._inverse[self.untile] = torch.arange(self.seq_len, device=payload.device)
        shape = (3, self.rows, *payload.shape[3:])
        if self._buf is None or self._buf.shape != shape:
            self._buf = torch.empty(shape, dtype=torch.bfloat16, device=payload.device)
        return self._inverse, self._buf

    def tiles_from_fp8(self, payload: torch.Tensor, scale: torch.Tensor, *, query_start: int | None = None):
        from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_tiles

        inverse, output = self.fp8_buffers(payload)
        return unpack_fp8_tiles(payload, scale, inverse, output, query_start=query_start)


def vsa_fp4_attention_sp(attn: Any, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor],
                         meta: MiniMaxH3VSAMetadata, use_fused_rope: bool, sp_group: Any,
                         pre_quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Ulysses-SP attention core on local sequence rows ``[1, rows, C]``; returns pre-``to_out`` ``[1, rows, H*D]``."""
    import torch.distributed as dist

    api = _api()
    world, rank = sp_group.world_size, sp_group.rank_in_group
    heads, dim = attn.num_attention_heads, attn.attention_head_dim
    local_rows = hidden_states.shape[1]
    chunks = getattr(attn, "_vsa_fp4_options", {}).get("sp_head_chunks", 1)
    if heads % (world * chunks):
        raise ValueError("H3 SP attention heads must divide evenly across ranks and head chunks")
    if (chunks > 1 and use_fused_rope and hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
            and attn.to_gate_compress is not None and attn._gate_active()):
        from fastvideo.models.dits.minimax_h3_fusions.sp_head_pipeline import attention_pipelined
        return attention_pipelined(attn, hidden_states, rotary_emb, meta, sp_group, chunks, pre_quantized=pre_quantized)
    layout = getattr(meta, "_h3_fp4_sp_layout", None)
    if layout is None:
        layout = _SPTileLayout(meta, rank, local_rows)
        meta._h3_fp4_sp_layout = layout  # type: ignore[attr-defined]

    with STAGES.span("qkv_proj_rope"):
        shared = _shared_qkv_gate_projections(attn, hidden_states, pre_quantized=pre_quantized)
        if shared is None:
            projected = _shared_input_projections((attn.to_q, attn.to_k, attn.to_v), hidden_states)
            gate_pre = None
        else:
            projected, gate_pre = shared
        query, key, value = (t.unflatten(-1, (heads, dim)) for t in projected)
        del projected, shared
        fused_transport = (use_fused_rope and query.is_cuda and query.dtype == torch.bfloat16
                           and all(t.is_contiguous() for t in (query, key, value)))
        if use_fused_rope:
            cos, sin = rotary_emb[0].to(query.dtype), rotary_emb[1].to(query.dtype)
            if not fused_transport:
                from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
                query = fused_qknorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
                key = fused_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
        else:
            query = attn._apply_rotary_emb(attn.norm_q(query), rotary_emb)
            key = attn._apply_rotary_emb(attn.norm_k(key), rotary_emb)

    gate = None

    def project_gate() -> None:
        nonlocal gate
        with STAGES.span("gate_proj"):
            if gate_pre is None:
                gate, _ = attn.to_gate_compress(hidden_states)
            else:
                linear = attn.to_gate_compress
                gate = linear.quant_method.apply(linear, hidden_states, linear.bias, pre_quantized=gate_pre)

    gate_active = attn.to_gate_compress is not None and attn._gate_active()
    with STAGES.span("qkv_pack"):
        if fused_transport:
            from fastvideo.models.dits.minimax_h3_fusions.sp_qkv_pack import pack_normalized_qkv_fp8
            payload, scale = pack_normalized_qkv_fp8(
                query[0], key[0], value[0], attn.norm_q.weight, attn.norm_k.weight,
                cos, sin, attn.norm_q.eps, attn.norm_k.eps, world,
            )
        else:
            payload, scale = _pack_heads_fp8(query[0], key[0], value[0], world)
    with STAGES.span("qkv_all_to_all"):
        payload, scale = _all_to_all(payload, scale, sp_group.device_group,
                                     overlap=project_gate if gate_active else None)
    with STAGES.span("qkv_unpack_tile"):
        qkv = layout.tiles_from_fp8(payload, scale)  # [3, R, Hs, D]
    q_t, k_t, v_t = qkv[0:1], qkv[1:2], qkv[2:3]

    vbs = meta.variable_block_sizes
    logical = layout.n_tiles * layout.tile
    prefix_mode, prefix_rows = _prefix_settings(attn, meta, layout.n_tiles)
    with STAGES.span("select_mask"):
        scores = torch.matmul(_pool_tiles(q_t[:, :logical], vbs, layout.tile),
                              _pool_tiles(k_t[:, :logical], vbs, layout.tile).transpose(-2, -1)) / (dim**0.5)
        sparsity = 0.0 if attn._layer_idx in meta.dense_layers else meta.VSA_sparsity
        mask = _build_block_mask(scores, meta.num_prefix_tiles, sparsity, meta.exempt, meta.video_tile_spans,
                                 meta.span_sparsities)
        q2k_idx, q2k_num, kv_valid, q2k_quad = _tile_mask_to_fp4_blocks(api, mask, layout.tile, vbs)
    with STAGES.span("fp4_attention"):
        out_bhsd = _fp4_fine_attention(api, q_t, k_t, v_t, q2k_idx, q2k_num, kv_valid, q2k_quad,
                                       logical=logical, prefix_rows=prefix_rows, precision=prefix_mode,
                                       options=getattr(attn, "_vsa_fp4_options", {}), layout=layout)

    out_c, gathered = None, None

    def compress_gate() -> None:
        nonlocal out_c, gathered
        with STAGES.span("gate_compress"):
            v_pooled = _pool_tiles(v_t[:, :logical], vbs, layout.tile)
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)[0].to(qkv.dtype)  # [Hs, n_tiles, D]
            gathered = torch.empty((world, *out_c.shape), dtype=out_c.dtype, device=out_c.device)

    with STAGES.span("out_pack"):
        payload, scale = _pack_seq_fp8(out_bhsd, layout.untile, world, local_rows)
    with STAGES.span("out_all_to_all"):
        payload, scale = _all_to_all(payload, scale, sp_group.device_group,
                                     overlap=compress_gate if gate_active else None)
    if gate_active:
        assert out_c is not None and gathered is not None
        with STAGES.span("gate_all_gather"):
            dist.all_gather_into_tensor(gathered, out_c.contiguous(), group=sp_group.device_group)
        with STAGES.span("gate_apply"):
            from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_gate
            from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

            assert gate is not None
            local_gate = gate[0].unflatten(-1, (heads, dim))
            method = getattr(attn.to_out, "quant_method", None)
            if (type(method) is NVFP4QuantizeMethod and method.wants_prequantized_input()
                    and method.uses_unit_activation_scale(attn.to_out) and gate.shape[-1] % 64 == 0):
                prepared = unpack_fp8_gate(payload, scale, gathered, local_gate, layout.local_row_tile, method.x_global_sf)
                return gate, prepared
            out = unpack_fp8_gate(payload, scale, gathered, local_gate, layout.local_row_tile)
    else:
        with STAGES.span("out_unpack"):
            out = _unpack_heads_fp8(payload, scale)  # [rows, H, D]
    return out.flatten(1, 2).unsqueeze(0)
