# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-TI2V-5B dense MLX runtime — Track D.

The Wan2.2 TI2V-5B (FullAttn) differs from the ported Wan2.1-T2V only in:

- **Scale** (24 heads x 128, hidden 3072, ffn 14336) — pure config, block math
  identical, so the dense loader ``mlx_dit_from_diffusers_safetensors`` loads the
  weights unchanged and we re-wrap the blocks here.
- **Per-token timestep conditioning** (``expand_timesteps=True``): the timestep is
  ``[batch, seq_len]`` (a level per patch token — how TI2V keeps the conditioning
  image frame at t=0 while the video frames are noised). ``timestep_proj`` becomes
  ``[batch, seq_len, 6, dim]`` and the block/output modulation is per-token
  (``[B, L, dim]``), a direct broadcast — this module implements exactly that.

I2V rides on the same forward: encode the image, replace the first latent frame,
and set that frame's timestep to 0 (handled by the caller / sampler). See
``docs/design/ti2v_5b_port_guide.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastvideo.mlx_runtime.fastwan import (
    MLXWanT2VCrossAttention,
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


class MLXWan22TransformerBlock:
    """Dense Wan block with per-token (``[B, L, dim]``) timestep modulation."""

    def __init__(self, weights: dict[str, mx.array], *, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
        self.weights = weights
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.attn2 = MLXWanT2VCrossAttention(weights, dim=dim, num_heads=num_heads, eps=eps)

    def __call__(self, hidden_states, encoder_hidden_states, timestep_proj, cos, sin) -> mx.array:
        import mlx.core as mx

        orig_dtype = hidden_states.dtype
        batch = hidden_states.shape[0]

        # timestep_proj: [B, L, 6, dim] -> six per-token [B, L, dim] modulations.
        e = self.weights["scale_shift_table"][None] + timestep_proj.astype(mx.float32)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [
            part.squeeze(2) for part in mx.split(e, 6, axis=2)
        ]

        # 1. Self-attention (dense, bidirectional) with per-token modulation.
        norm_hidden = layer_norm(hidden_states.astype(mx.float32), eps=self.eps)
        norm_hidden = (norm_hidden * (1.0 + scale_msa) + shift_msa).astype(orig_dtype)

        query = linear(norm_hidden, self.weights["to_q.weight"], self.weights.get("to_q.bias"))
        key = linear(norm_hidden, self.weights["to_k.weight"], self.weights.get("to_k.bias"))
        value = linear(norm_hidden, self.weights["to_v.weight"], self.weights.get("to_v.bias"))

        query = rms_norm(query, self.weights["norm_q.weight"],
                         eps=self.eps).reshape(batch, -1, self.num_heads, self.head_dim)
        key = rms_norm(key, self.weights["norm_k.weight"], eps=self.eps).reshape(batch, -1, self.num_heads,
                                                                                 self.head_dim)
        value = value.reshape(batch, -1, self.num_heads, self.head_dim)

        from fastvideo.mlx_runtime.fastwan import apply_rotary_emb
        query = apply_rotary_emb(query, cos, sin, is_neox_style=False)
        key = apply_rotary_emb(key, cos, sin, is_neox_style=False)

        attn = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
        ).transpose(0, 2, 1, 3)
        attn = attn.reshape(batch, -1, self.dim)
        attn = linear(attn, self.weights["to_out.weight"], self.weights.get("to_out.bias"))

        hidden_states = hidden_states + (attn * gate_msa).astype(orig_dtype)
        norm_hidden = layer_norm(hidden_states.astype(mx.float32),
                                 weight=self.weights["self_attn_residual_norm.norm.weight"],
                                 bias=self.weights["self_attn_residual_norm.norm.bias"],
                                 eps=self.eps).astype(orig_dtype)

        # 2. Cross-attention, then per-token shift/scale modulation.
        cross = self.attn2(norm_hidden, encoder_hidden_states)
        hidden_states = hidden_states + cross
        norm_hidden = layer_norm(hidden_states.astype(mx.float32), eps=self.eps)
        norm_hidden = (norm_hidden * (1.0 + c_scale_msa) + c_shift_msa).astype(orig_dtype)

        # 3. Feed-forward with per-token gate.
        ff = linear(norm_hidden, self.weights["ffn.fc_in.weight"], self.weights.get("ffn.fc_in.bias"))
        ff = gelu_tanh(ff)
        ff = linear(ff, self.weights["ffn.fc_out.weight"], self.weights.get("ffn.fc_out.bias"))
        hidden_states = hidden_states + (ff * c_gate_msa).astype(orig_dtype)
        return hidden_states.astype(orig_dtype)


class MLXWan22DiT:
    """Wan2.2-TI2V-5B dense DiT with per-token timestep conditioning."""

    def __init__(self, weights: dict[str, mx.array], blocks: list[MLXWan22TransformerBlock], config: dict) -> None:
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

    def _patch_embed(self, hidden_states) -> mx.array:
        batch, channels, frames, height, width = hidden_states.shape
        pt, ph, pw = self.patch_size
        patch_dim = channels * pt * ph * pw
        x = hidden_states.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, -1, patch_dim)
        return linear(x, self.weights["patch_embedding.weight"], self.weights.get("patch_embedding.bias"))

    def _condition(self, timestep, encoder_hidden_states) -> tuple:
        """Per-token conditioning. ``timestep`` is ``[B, L]`` (one level per token)."""
        batch, seq = timestep.shape
        t_freq = timestep_embedding(timestep.reshape(-1), self.freq_dim).astype(
            weight_dtype(self.weights["condition_embedder.time_embedder.linear_1.weight"]))
        temb = linear(t_freq, self.weights["condition_embedder.time_embedder.linear_1.weight"],
                      self.weights["condition_embedder.time_embedder.linear_1.bias"])
        temb = silu(temb)
        temb = linear(temb, self.weights["condition_embedder.time_embedder.linear_2.weight"],
                      self.weights["condition_embedder.time_embedder.linear_2.bias"])
        timestep_proj = linear(silu(temb), self.weights["condition_embedder.time_proj.weight"],
                               self.weights["condition_embedder.time_proj.bias"])
        timestep_proj = timestep_proj.reshape(batch, seq, 6, self.hidden_size)

        ehs = linear(encoder_hidden_states, self.weights["condition_embedder.text_embedder.linear_1.weight"],
                     self.weights["condition_embedder.text_embedder.linear_1.bias"])
        ehs = gelu_tanh(ehs)
        ehs = linear(ehs, self.weights["condition_embedder.text_embedder.linear_2.weight"],
                     self.weights["condition_embedder.text_embedder.linear_2.bias"])
        temb_out = temb.reshape(batch, seq, self.hidden_size)
        return temb_out, timestep_proj, ehs

    def _output(self, hidden_states, temb_out, *, batch, frames, height, width) -> mx.array:
        import mlx.core as mx

        pt, ph, pw = self.patch_size
        post_pt, post_ph, post_pw = frames // pt, height // ph, width // pw
        # Per-token output modulation: scale_shift_table[1,2,dim] + temb[B,L,1,dim].
        e = self.weights["scale_shift_table"][None] + temb_out[:, :, None, :].astype(mx.float32)
        shift, scale = [part.squeeze(2) for part in mx.split(e, 2, axis=2)]
        norm = layer_norm(hidden_states.astype(mx.float32), eps=self.eps)
        norm = (norm * (1.0 + scale) + shift).astype(weight_dtype(self.weights["proj_out.weight"]))
        out = linear(norm, self.weights["proj_out.weight"], self.weights["proj_out.bias"])
        out = out.reshape(batch, post_pt, post_ph, post_pw, pt, ph, pw, self.out_channels)
        out = out.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        return out.reshape(batch, self.out_channels, frames, height, width)

    def __call__(self, hidden_states, encoder_hidden_states, timestep, freqs_cis) -> mx.array:
        cos, sin = freqs_cis
        batch, _, frames, height, width = hidden_states.shape
        hidden = self._patch_embed(hidden_states)
        temb_out, timestep_proj, ehs = self._condition(timestep, encoder_hidden_states)
        for block in self.blocks:
            hidden = block(hidden, ehs, timestep_proj, cos, sin)
        return self._output(hidden, temb_out, batch=batch, frames=frames, height=height, width=width)


def mlx_wan22_dit_from_diffusers_safetensors(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    dtype: str = "fp16",
    num_blocks: int | None = None,
    quantization=None,
) -> MLXWan22DiT:
    """Load Wan2.2-TI2V-5B (FullAttn) into ``MLXWan22DiT`` via the dense loader."""
    dense = mlx_dit_from_diffusers_safetensors(checkpoint_path,
                                               config_path,
                                               dtype=dtype,
                                               num_blocks=num_blocks,
                                               quantization=quantization)
    inner_dim = int(dense.config["num_attention_heads"]) * int(dense.config["attention_head_dim"])
    blocks = [
        MLXWan22TransformerBlock(block.weights,
                                 dim=inner_dim,
                                 ffn_dim=int(dense.config["ffn_dim"]),
                                 num_heads=int(dense.config["num_attention_heads"]),
                                 eps=float(dense.config.get("eps", 1e-6))) for block in dense.blocks
    ]
    return MLXWan22DiT(dense.weights, blocks, dense.config)
