# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""Target-rows-only DiT forward over a recorded prefix K/V cache."""

from __future__ import annotations

from typing import Any

import numpy as np

from fastvideo.mlx_runtime.fastwan import weight_dtype
from fastvideo.mlx_runtime.minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    _feed_forward,
    _h3_rms_norm,
    _modulate,
    apply_h3_rotary,
    linear,
    rope_cos_sin,
)
from fastvideo.mlx_runtime.minimax_h3_prefix_kv import (
    MiniMaxH3PrefixKVCache,
    build_target_query_plan,
    frozen_prefix_rows,
)
from fastvideo.mlx_runtime.minimax_h3_vsa import MiniMaxH3VSAGeometry, MiniMaxH3VSAStats
from fastvideo.mlx_runtime.minimax_h3_vsa_target import h3_vsa_attention_target


def _attention_target(
    weights: dict[str, Any],
    x,
    cos,
    sin,
    k_prefix,
    v_prefix,
    *,
    num_heads: int,
    head_dim: int,
    eps: float,
    vsa_geometry: MiniMaxH3VSAGeometry | None,
    vsa_plan: Any,
    vsa_sparsity: float,
    vsa_exempt: bool,
    vsa_ref_keep_rate: float | None,
    vsa_impl: str,
    vsa_stats: MiniMaxH3VSAStats | None,
    use_gate_compress: bool,
):
    """Attention for target rows: fresh target Q, cached-prefix + fresh-target K/V."""
    import mlx.core as mx

    rows = x.shape[0]
    q = linear(x, weights["attn.to_q.weight"]).reshape(rows, num_heads, head_dim)
    k = linear(x, weights["attn.to_k.weight"]).reshape(rows, num_heads, head_dim)
    v = linear(x, weights["attn.to_v.weight"]).reshape(rows, num_heads, head_dim)
    q = apply_h3_rotary(_h3_rms_norm(q, weights["attn.norm_q.weight"], eps), cos, sin)
    k = apply_h3_rotary(_h3_rms_norm(k, weights["attn.norm_k.weight"], eps), cos, sin)
    key = mx.concatenate([k_prefix.astype(k.dtype), k], axis=0)
    value = mx.concatenate([v_prefix.astype(v.dtype), v], axis=0)
    gate = None
    if use_gate_compress and "attn.to_gate_compress.weight" in weights:
        gate = linear(x, weights["attn.to_gate_compress.weight"]).reshape(rows, num_heads, head_dim)
    if vsa_geometry is not None and (vsa_sparsity > 0.0 or gate is not None):
        attended = h3_vsa_attention_target(
            q,
            key,
            value,
            vsa_geometry,
            vsa_plan,
            sparsity=vsa_sparsity,
            exempt=vsa_exempt,
            gate_compress=gate,
            impl=vsa_impl,  # type: ignore[arg-type]
            stats=vsa_stats,
            ref_keep_rate=vsa_ref_keep_rate,
        )
    else:
        if vsa_stats is not None:
            vsa_stats.impl = "dense"
            if vsa_geometry is not None:
                vsa_stats.num_prefix_tiles = vsa_geometry.num_prefix_tiles
                vsa_stats.num_video_tiles = vsa_geometry.num_video_tiles
                vsa_stats.video_keep = vsa_geometry.num_video_tiles
        attended = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            key.transpose(1, 0, 2)[None],
            value.transpose(1, 0, 2)[None],
            scale=head_dim**-0.5,
        )[0].transpose(1, 0, 2)
    return linear(attended.reshape(rows, num_heads * head_dim), weights["attn.to_out.0.weight"])


def _transformer_block_target(
    weights: dict[str, Any],
    hidden_states,
    tables,
    adaln_indices,
    cos,
    sin,
    k_prefix,
    v_prefix,
    *,
    norm_eps: float,
    **attention_kwargs: Any,
):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = tables
    normed = _h3_rms_norm(hidden_states, weights["norm1.weight"], norm_eps)
    normed = _modulate(normed, (1.0 + scale_msa)[adaln_indices], shift_msa[adaln_indices])
    attn_output = _attention_target(
        weights,
        normed.astype(weight_dtype(weights["attn.to_q.weight"])),
        cos,
        sin,
        k_prefix,
        v_prefix,
        **attention_kwargs,
    )
    hidden_states = hidden_states + gate_msa[adaln_indices] * attn_output
    normed = _h3_rms_norm(hidden_states, weights["norm2.weight"], norm_eps)
    normed = _modulate(normed, (1.0 + scale_mlp)[adaln_indices], shift_mlp[adaln_indices])
    ff_output = _feed_forward(weights, normed.astype(weight_dtype(weights["ff.net.0.proj.weight"])))
    return hidden_states + gate_mlp[adaln_indices] * ff_output


def forward_target_rows(
    dit,
    video_target_rows,
    audio_target_rows,
    *,
    layout,
    step_timesteps: np.ndarray,
    row_timestep_inverse: np.ndarray,
    prefix_kv: MiniMaxH3PrefixKVCache,
    step_index: int = 0,
    pdd_block: tuple[int, int] | None = None,
    pdd_integration_weights: dict[str, np.ndarray] | None = None,
):
    """One denoise step on the target audio + video rows, attending to ``prefix_kv``.

    ``video_target_rows``/``audio_target_rows`` are the noisy target rows
    alone (no condition rows). Returns ``(video_output, audio_output)`` for
    those rows; the frozen text/reference rows are neither embedded, run
    through any block, nor projected.
    """
    import mlx.core as mx

    if dit._adaln_cache is None:
        raise RuntimeError("precompute_adaln() must run before forward_target_rows().")
    frozen = frozen_prefix_rows(layout)
    if not prefix_kv.complete or prefix_kv.rows != frozen or prefix_kv.num_layers != len(dit.blocks):
        raise ValueError("The prefix K/V cache is incomplete or was recorded for a different packed layout.")
    num_audio = int(layout.audio_indices.shape[0]) - int(layout.num_condition_audio_rows)
    num_video = int(layout.video_indices.shape[0]) - int(layout.num_condition_video_rows)
    if video_target_rows.shape[0] != num_video or audio_target_rows.shape[0] != num_audio:
        raise ValueError(f"Expected {num_video} video / {num_audio} audio target rows, got "
                         f"{video_target_rows.shape[0]} / {audio_target_rows.shape[0]}.")
    geometry = dit.prepare_vsa_geometry(layout) if dit.vsa_config.enabled else None
    plan = build_target_query_plan(geometry, frozen, num_audio, num_video) if geometry is not None else None

    cache = dit._adaln_cache
    positions = mx.array(cache.positions(step_timesteps))
    cos, sin = rope_cos_sin(mx.array(layout.position_ids[frozen:]), dit.rope_freq_dim, dit.rope_theta)
    video_embeds = linear(
        video_target_rows.astype(weight_dtype(dit.weights["proj_in.weight"])),
        dit.weights["proj_in.weight"],
        dit.weights["proj_in.bias"],
    )
    audio_embeds = linear(
        audio_target_rows.astype(weight_dtype(dit.weights["audio_proj_in.weight"])),
        dit.weights["audio_proj_in.weight"],
        dit.weights["audio_proj_in.bias"],
    )
    packed = mx.concatenate([audio_embeds, video_embeds.astype(audio_embeds.dtype)], axis=0)
    if prefix_kv.hidden_dtype is not None:
        packed = packed.astype(prefix_kv.hidden_dtype)

    row_positions = positions[mx.array(row_timestep_inverse[frozen:])]
    token_tags = mx.array(layout.token_tags[frozen:])
    adaln_indices = (row_positions * MINIMAX_H3_MODALITY_NUM + token_tags).astype(mx.int32)

    for block_index, (block, tables) in enumerate(zip(dit.blocks, cache.block_tables, strict=True)):
        vsa_kwargs = dit._vsa_block_kwargs(block_index, step_index)
        k_prefix, v_prefix = prefix_kv.load(block_index)
        packed = _transformer_block_target(
            block,
            packed,
            tables,
            adaln_indices,
            cos,
            sin,
            k_prefix,
            v_prefix,
            norm_eps=dit.norm_eps,
            num_heads=dit.num_heads,
            head_dim=dit.head_dim,
            eps=dit.qk_norm_eps,
            vsa_geometry=vsa_kwargs["vsa_geometry"],
            vsa_plan=plan,
            vsa_sparsity=vsa_kwargs["vsa_sparsity"],
            vsa_exempt=vsa_kwargs["vsa_exempt"],
            vsa_ref_keep_rate=vsa_kwargs["vsa_ref_keep_rate"],
            vsa_impl=vsa_kwargs["vsa_impl"],
            vsa_stats=vsa_kwargs["vsa_stats"],
            use_gate_compress=vsa_kwargs["use_gate_compress"],
        )
        mx.eval(packed)
        if dit.last_vsa_stats is not None:
            dit.last_vsa_stats.record(vsa_kwargs["vsa_stats"])

    normed = _h3_rms_norm(packed, dit.weights["norm_out.norm.weight"], dit.final_norm_eps)
    normed = normed * (1.0 + cache.norm_out_scale[row_positions]) + cache.norm_out_shift[row_positions]
    video_weight, video_bias = dit._fused_output_params("proj_out.weight", "video", pdd_block, pdd_integration_weights)
    audio_weight, audio_bias = dit._fused_output_params("audio_proj_out.weight", "audio", pdd_block,
                                                        pdd_integration_weights)
    video_output = linear(normed[num_audio:].astype(weight_dtype(video_weight)), video_weight, video_bias)
    audio_output = linear(normed[:num_audio].astype(weight_dtype(audio_weight)), audio_weight, audio_bias)
    return video_output, audio_output
