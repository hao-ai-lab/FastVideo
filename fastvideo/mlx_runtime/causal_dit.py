# SPDX-License-Identifier: Apache-2.0
"""Causal (streaming) Wan DiT forward for the MLX runtime — Track C, rung 4.

Ports ``CausalWanTransformer3DModel._forward_inference`` (the KV-cached,
chunk-at-a-time path from ``fastvideo/models/dits/causal_wanvideo.py``) to MLX.
Reuses the dense port's patch-embed / condition / output / block weights
(``fastvideo.mlx_runtime.fastwan``) unchanged — the causal checkpoint has the
same weight layout — and swaps the self-attention for the cached, mask-free
``causal_self_attention_step`` plus a per-generation cross-attention cache and
per-frame timestep modulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastvideo.mlx_runtime.causal import MLXCausalKVCache, causal_self_attention_step
from fastvideo.mlx_runtime.fastwan import (
    gelu_tanh,
    layer_norm,
    linear,
    mlx_dit_from_diffusers_safetensors,
    rms_norm,
    silu,
    timestep_embedding,
    weight_dtype,
)

if TYPE_CHECKING:
    import mlx.core as mx


def _modulate_per_frame(x, scale, shift, *, temb_seq_len: int, tokens_per_temb: int):
    """``(x reshaped to per-frame) * (1 + scale) + shift`` then flattened back.

    Mirrors the torch block's ``unflatten(1,(S,tpt)) * (1+scale) + shift`` so the
    modulation is applied per timestep-frame (``scale``/``shift`` are
    ``[B, S, 1, dim]``).
    """
    batch, seq, dim = x.shape
    x = x.reshape(batch, temb_seq_len, tokens_per_temb, dim)
    x = x * (1.0 + scale) + shift
    return x.reshape(batch, seq, dim)


def _gate_per_frame(x, gate, *, temb_seq_len: int, tokens_per_temb: int):
    batch, seq, dim = x.shape
    x = x.reshape(batch, temb_seq_len, tokens_per_temb, dim)
    x = x * gate
    return x.reshape(batch, seq, dim)


class MLXCausalWanTransformerBlock:
    """Causal Wan block: cached self-attention + cross-attention cache."""

    def __init__(self, weights: dict[str, mx.array], *, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
        self.weights = weights
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

    def _cross_attention(self, x, encoder_hidden_states, crossattn_cache) -> mx.array:
        import mlx.core as mx

        batch = x.shape[0]
        q = linear(x, self.weights["attn2.to_q.weight"], self.weights.get("attn2.to_q.bias"))
        q = rms_norm(q, self.weights["attn2.norm_q.weight"], eps=self.eps).reshape(batch, -1, self.num_heads,
                                                                                   self.head_dim)

        if crossattn_cache is not None and crossattn_cache.get("is_init"):
            key = crossattn_cache["k"]
            value = crossattn_cache["v"]
        else:
            key = linear(encoder_hidden_states, self.weights["attn2.to_k.weight"], self.weights.get("attn2.to_k.bias"))
            key = rms_norm(key, self.weights["attn2.norm_k.weight"],
                           eps=self.eps).reshape(batch, -1, self.num_heads, self.head_dim)
            value = linear(encoder_hidden_states, self.weights["attn2.to_v.weight"],
                           self.weights.get("attn2.to_v.bias")).reshape(batch, -1, self.num_heads, self.head_dim)
            if crossattn_cache is not None:
                crossattn_cache["k"] = key
                crossattn_cache["v"] = value
                crossattn_cache["is_init"] = True

        attended = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
        ).transpose(0, 2, 1, 3)
        attended = attended.reshape(batch, -1, self.dim)
        return linear(attended, self.weights["attn2.to_out.weight"], self.weights.get("attn2.to_out.bias"))

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        cos,
        sin,
        *,
        kv_cache: MLXCausalKVCache,
        crossattn_cache,
        current_start: int,
        local_attn_size: int,
        frame_seqlen: int,
    ) -> mx.array:
        import mlx.core as mx

        orig_dtype = hidden_states.dtype
        batch, seq_length, _ = hidden_states.shape
        temb_seq_len = timestep_proj.shape[1]
        tokens_per_temb = seq_length // temb_seq_len

        # e = scale_shift_table[1,6,dim] + timestep_proj[B,S,6,dim] -> [B,S,6,dim]
        e = self.weights["scale_shift_table"][None] + timestep_proj.astype(mx.float32)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [
            part.reshape(batch, temb_seq_len, 1, self.dim) for part in mx.split(e, 6, axis=2)
        ]

        # 1. Self-attention with per-frame modulation.
        norm_hidden = layer_norm(hidden_states.astype(mx.float32), eps=self.eps)
        norm_hidden = _modulate_per_frame(norm_hidden,
                                          scale_msa,
                                          shift_msa,
                                          temb_seq_len=temb_seq_len,
                                          tokens_per_temb=tokens_per_temb)
        norm_hidden = norm_hidden.astype(orig_dtype)

        query = linear(norm_hidden, self.weights["to_q.weight"], self.weights.get("to_q.bias"))
        key = linear(norm_hidden, self.weights["to_k.weight"], self.weights.get("to_k.bias"))
        value = linear(norm_hidden, self.weights["to_v.weight"], self.weights.get("to_v.bias"))

        query = rms_norm(query, self.weights["norm_q.weight"],
                         eps=self.eps).reshape(batch, -1, self.num_heads, self.head_dim)
        key = rms_norm(key, self.weights["norm_k.weight"], eps=self.eps).reshape(batch, -1, self.num_heads,
                                                                                 self.head_dim)
        value = value.reshape(batch, -1, self.num_heads, self.head_dim)

        attn = causal_self_attention_step(query,
                                          key,
                                          value,
                                          cos,
                                          sin,
                                          kv_cache,
                                          current_start=current_start,
                                          local_attn_size=local_attn_size,
                                          frame_seqlen=frame_seqlen)
        attn = attn.reshape(batch, -1, self.dim)
        attn = linear(attn, self.weights["to_out.weight"], self.weights.get("to_out.bias"))

        # Residual (per-frame gate) + norm; self-attn residual-norm has null shift/scale.
        residual = hidden_states + _gate_per_frame(
            attn, gate_msa, temb_seq_len=temb_seq_len, tokens_per_temb=tokens_per_temb)
        norm_hidden = layer_norm(residual.astype(mx.float32),
                                 weight=self.weights["self_attn_residual_norm.norm.weight"],
                                 bias=self.weights["self_attn_residual_norm.norm.bias"],
                                 eps=self.eps).astype(orig_dtype)
        hidden_states = residual.astype(orig_dtype)

        # 2. Cross-attention (cached), then residual-norm with per-frame shift/scale.
        cross = self._cross_attention(norm_hidden, encoder_hidden_states, crossattn_cache)
        residual = hidden_states + cross
        norm_hidden = layer_norm(residual.astype(mx.float32), eps=self.eps)
        norm_hidden = _modulate_per_frame(norm_hidden,
                                          c_scale_msa,
                                          c_shift_msa,
                                          temb_seq_len=temb_seq_len,
                                          tokens_per_temb=tokens_per_temb)
        norm_hidden = norm_hidden.astype(orig_dtype)
        hidden_states = residual.astype(orig_dtype)

        # 3. Feed-forward with per-frame gate.
        ff = linear(norm_hidden, self.weights["ffn.fc_in.weight"], self.weights.get("ffn.fc_in.bias"))
        ff = gelu_tanh(ff)
        ff = linear(ff, self.weights["ffn.fc_out.weight"], self.weights.get("ffn.fc_out.bias"))
        hidden_states = hidden_states + _gate_per_frame(
            ff, c_gate_msa, temb_seq_len=temb_seq_len, tokens_per_temb=tokens_per_temb)
        return hidden_states.astype(orig_dtype)


class MLXCausalWanDiT:
    """Causal Wan DiT with KV-cached, chunk-at-a-time inference (Track C)."""

    def __init__(
        self,
        weights: dict[str, mx.array],
        blocks: list[MLXCausalWanTransformerBlock],
        config: dict,
        *,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frames_per_block: int = 1,
    ) -> None:
        self.weights = weights
        self.blocks = blocks
        self.config = config
        self.num_heads = int(config["num_attention_heads"])
        self.head_dim = int(config["attention_head_dim"])
        self.hidden_size = self.num_heads * self.head_dim
        self.freq_dim = int(config["freq_dim"])
        self.patch_size = tuple(config["patch_size"])
        self.out_channels = int(config["out_channels"])
        self.eps = float(config.get("eps", 1e-6))
        self.text_len = int(config.get("text_len", 512))
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frames_per_block = num_frames_per_block

    def allocate_caches(self, *, batch: int, frame_seqlen: int, dtype=None) -> tuple:
        """One KV cache + cross-attn cache per block, sized to the window."""
        from fastvideo.mlx_runtime.causal import max_attention_size

        window = max_attention_size(self.local_attn_size, frame_seqlen)
        kv_caches = [
            MLXCausalKVCache.allocate(batch=batch,
                                      max_tokens=window,
                                      num_heads=self.num_heads,
                                      head_dim=self.head_dim,
                                      sink_tokens=self.sink_size * frame_seqlen,
                                      dtype=dtype) for _ in self.blocks
        ]
        crossattn_caches: list[dict] = [{"is_init": False} for _ in self.blocks]
        return kv_caches, crossattn_caches

    def _patch_embed(self, hidden_states) -> mx.array:
        batch, channels, frames, height, width = hidden_states.shape
        pt, ph, pw = self.patch_size
        patch_dim = channels * pt * ph * pw
        x = hidden_states.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, -1, patch_dim)
        return linear(x, self.weights["patch_embedding.weight"], self.weights.get("patch_embedding.bias"))

    def _condition(self, timestep, encoder_hidden_states) -> tuple:
        """Per-frame timestep conditioning. ``timestep`` is ``[B, frames]``."""
        import mlx.core as mx

        batch, frames = timestep.shape
        t_freq = timestep_embedding(timestep.reshape(-1), self.freq_dim).astype(
            weight_dtype(self.weights["condition_embedder.time_embedder.linear_1.weight"]))
        temb = linear(t_freq, self.weights["condition_embedder.time_embedder.linear_1.weight"],
                      self.weights["condition_embedder.time_embedder.linear_1.bias"])
        temb = silu(temb)
        temb = linear(temb, self.weights["condition_embedder.time_embedder.linear_2.weight"],
                      self.weights["condition_embedder.time_embedder.linear_2.bias"])
        timestep_proj = linear(silu(temb), self.weights["condition_embedder.time_proj.weight"],
                               self.weights["condition_embedder.time_proj.bias"])
        timestep_proj = timestep_proj.reshape(batch, frames, 6, self.hidden_size)

        # Pad the text sequence to text_len with zeros, exactly like the torch
        # causal model does before the text embedder (causal_wanvideo.py).
        pad = self.text_len - encoder_hidden_states.shape[1]
        if pad > 0:
            encoder_hidden_states = mx.concatenate([
                encoder_hidden_states,
                mx.zeros((encoder_hidden_states.shape[0], pad, encoder_hidden_states.shape[2]),
                         dtype=encoder_hidden_states.dtype)
            ],
                                                   axis=1)

        ehs = linear(encoder_hidden_states, self.weights["condition_embedder.text_embedder.linear_1.weight"],
                     self.weights["condition_embedder.text_embedder.linear_1.bias"])
        ehs = gelu_tanh(ehs)
        ehs = linear(ehs, self.weights["condition_embedder.text_embedder.linear_2.weight"],
                     self.weights["condition_embedder.text_embedder.linear_2.bias"])
        temb_out = temb.reshape(batch, frames, self.hidden_size)
        return temb_out, timestep_proj, ehs

    def _output(self, hidden_states, temb_out, *, batch, frames, height, width) -> mx.array:
        import mlx.core as mx

        pt, ph, pw = self.patch_size
        post_pt, post_ph, post_pw = frames // pt, height // ph, width // pw
        # Per-frame output modulation: scale_shift_table[1,2,dim] + temb[B,F,1,dim].
        e = self.weights["scale_shift_table"][None] + temb_out[:, :, None, :].astype(mx.float32)
        shift, scale = [part.reshape(batch, frames, 1, self.hidden_size) for part in mx.split(e, 2, axis=2)]
        norm = layer_norm(hidden_states.astype(mx.float32), eps=self.eps)
        tokens_per_frame = norm.shape[1] // frames
        norm = _modulate_per_frame(norm, scale, shift, temb_seq_len=frames, tokens_per_temb=tokens_per_frame)
        norm = norm.astype(weight_dtype(self.weights["proj_out.weight"]))
        out = linear(norm, self.weights["proj_out.weight"], self.weights["proj_out.bias"])
        out = out.reshape(batch, post_pt, post_ph, post_pw, pt, ph, pw, self.out_channels)
        out = out.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        return out.reshape(batch, self.out_channels, frames, height, width)

    def forward_chunk(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        cos,
        sin,
        kv_caches,
        crossattn_caches,
        *,
        current_start: int,
    ) -> mx.array:
        """Denoise one frame-block; ``cos``/``sin`` are its global-position rotary."""
        batch, _, frames, height, width = hidden_states.shape
        frame_seqlen = (height // self.patch_size[1]) * (width // self.patch_size[2])

        hidden = self._patch_embed(hidden_states)
        temb_out, timestep_proj, ehs = self._condition(timestep, encoder_hidden_states)

        for block, kv_cache, crossattn_cache in zip(self.blocks, kv_caches, crossattn_caches, strict=False):
            hidden = block(hidden,
                           ehs,
                           timestep_proj,
                           cos,
                           sin,
                           kv_cache=kv_cache,
                           crossattn_cache=crossattn_cache,
                           current_start=current_start,
                           local_attn_size=self.local_attn_size,
                           frame_seqlen=frame_seqlen)

        return self._output(hidden, temb_out, batch=batch, frames=frames, height=height, width=width)


def mlx_causal_dit_from_diffusers_safetensors(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    dtype: str = "fp16",
    num_blocks: int | None = None,
    quantization=None,
    local_attn_size: int = -1,
    sink_size: int = 0,
    num_frames_per_block: int = 1,
) -> MLXCausalWanDiT:
    """Load a causal Wan DiT from a Diffusers checkpoint into ``MLXCausalWanDiT``.

    Reuses the dense Diffusers loader (the causal checkpoint has the same weight
    layout) and re-wraps its blocks as causal blocks — only the forward differs.
    """
    dense = mlx_dit_from_diffusers_safetensors(checkpoint_path,
                                               config_path,
                                               dtype=dtype,
                                               num_blocks=num_blocks,
                                               quantization=quantization)
    inner_dim = int(dense.config["num_attention_heads"]) * int(dense.config["attention_head_dim"])
    blocks = [
        MLXCausalWanTransformerBlock(block.weights,
                                     dim=inner_dim,
                                     ffn_dim=int(dense.config["ffn_dim"]),
                                     num_heads=int(dense.config["num_attention_heads"]),
                                     eps=float(dense.config.get("eps", 1e-6))) for block in dense.blocks
    ]
    return MLXCausalWanDiT(dense.weights,
                           blocks,
                           dense.config,
                           local_attn_size=local_attn_size,
                           sink_size=sink_size,
                           num_frames_per_block=num_frames_per_block)
