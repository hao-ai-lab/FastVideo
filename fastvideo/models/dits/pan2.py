# SPDX-License-Identifier: Apache-2.0
"""PAN2 video transformer.

A mixture-of-transformers DiT: the video and text streams keep separate weights and meet only in one joint
attention per block, over ``[video; text]``. The text stream is modulated by a learned, timestep-free table.
Under sequence parallelism the video tokens are sharded and the text tokens are replicated on every rank.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.configs.models.dits.pan2 import PAN2VideoConfig
from fastvideo.distributed.communication_op import (sequence_model_parallel_all_gather_with_unpad,
                                                    sequence_model_parallel_shard)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_1d_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed, TimestepEmbedder
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


class PAN2SwiGLU(nn.Module):
    """SwiGLU MLP of the text refiner; the fused input projection stores ``[value; gate]``."""

    def __init__(self, dim: int, inner_dim: int, prefix: str = ""):
        super().__init__()
        self.fc_in = ReplicatedLinear(dim, 2 * inner_dim, bias=False, prefix=f"{prefix}.fc_in")
        self.fc_out = ReplicatedLinear(inner_dim, dim, bias=False, prefix=f"{prefix}.fc_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        value, gate = x.chunk(2, dim=-1)
        x, _ = self.fc_out(value * F.silu(gate))
        return x


class PAN2RefinerAttention(nn.Module):
    """Text-only self-attention of the refiner; the text is replicated on every sequence-parallel rank."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 eps: float,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = ReplicatedLinear(dim, 3 * num_heads * head_dim, bias=False, prefix=f"{prefix}.to_qkv")
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = ReplicatedLinear(num_heads * head_dim, dim, bias=False, prefix=f"{prefix}.to_out")
        self.attn = LocalAttention(num_heads=num_heads,
                                   head_size=head_dim,
                                   causal=False,
                                   supported_attention_backends=supported_attention_backends)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.unflatten(-1, (3, self.num_heads, -1)).unbind(2)
        query = self.norm_q(query).to(value.dtype)
        key = self.norm_k(key).to(value.dtype)
        hidden_states = self.attn(query, key, value)
        hidden_states, _ = self.to_out(hidden_states.flatten(2))
        return hidden_states


class PAN2TokenRefinerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 mlp_ratio: float,
                 eps: float,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = PAN2RefinerAttention(dim,
                                         num_heads,
                                         head_dim,
                                         eps,
                                         supported_attention_backends=supported_attention_backends,
                                         prefix=f"{prefix}.attn")
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff = PAN2SwiGLU(dim, int(dim * mlp_ratio), prefix=f"{prefix}.ff")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class PAN2AdaLayerNormZero(nn.Module):
    """Timestep modulation of the video stream: modulated attention input plus the gate/shift/scale for the rest."""

    def __init__(self, dim: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.linear = ReplicatedLinear(dim, 6 * dim, bias=True, prefix=f"{prefix}.linear")
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor,
                temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb, _ = self.linear(F.silu(temb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class PAN2AdaLayerNormOut(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.linear = ReplicatedLinear(dim, 2 * dim, bias=True, prefix=f"{prefix}.linear")
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        emb, _ = self.linear(F.silu(temb).to(x.dtype))
        scale, shift = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class PAN2Attention(nn.Module):
    """Joint attention over ``[video; text]`` with per-stream projections and Q/K RMSNorm.

    RoPE applies to the video tokens only; ``DistributedAttention`` applies it after its all-to-all, where each rank
    holds the full (unpadded) video sequence for its heads.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 eps: float,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = num_heads * head_dim

        self.to_qkv = ReplicatedLinear(dim, 3 * inner_dim, bias=True, prefix=f"{prefix}.to_qkv")
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = ReplicatedLinear(inner_dim, dim, bias=True, prefix=f"{prefix}.to_out")

        self.add_qkv_proj = ReplicatedLinear(dim, 3 * inner_dim, bias=True, prefix=f"{prefix}.add_qkv_proj")
        self.norm_added_q = RMSNorm(head_dim, eps=eps)
        self.norm_added_k = RMSNorm(head_dim, eps=eps)
        self.to_add_out = ReplicatedLinear(inner_dim, dim, bias=True, prefix=f"{prefix}.to_add_out")

        self.attn = DistributedAttention(num_heads=num_heads,
                                         head_size=head_dim,
                                         causal=False,
                                         supported_attention_backends=supported_attention_backends,
                                         prefix=f"{prefix}.attn")

    def _project(self, projection: ReplicatedLinear,
                 hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = projection(hidden_states)
        query, key, value = qkv.unflatten(-1, (3, self.num_heads, -1)).unbind(2)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value = self._project(self.to_qkv, hidden_states)
        query = self.norm_q(query).to(value.dtype)
        key = self.norm_k(key).to(value.dtype)

        encoder_query, encoder_key, encoder_value = self._project(self.add_qkv_proj, encoder_hidden_states)
        encoder_query = self.norm_added_q(encoder_query).to(encoder_value.dtype)
        encoder_key = self.norm_added_k(encoder_key).to(encoder_value.dtype)

        hidden_states, encoder_hidden_states = self.attn(query,
                                                         key,
                                                         value,
                                                         original_seq_len,
                                                         replicated_q=encoder_query,
                                                         replicated_k=encoder_key,
                                                         replicated_v=encoder_value,
                                                         freqs_cis=freqs_cis)
        hidden_states, _ = self.to_out(hidden_states.flatten(2))
        encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states.flatten(2))
        return hidden_states, encoder_hidden_states


class PAN2TransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 mlp_ratio: float,
                 context_mlp_ratio: float,
                 eps: float = 1e-6,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.norm1 = PAN2AdaLayerNormZero(dim, eps=eps, prefix=f"{prefix}.norm1")
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.scale_shift_table_context = nn.Parameter(torch.zeros(6 * dim))

        self.attn = PAN2Attention(dim,
                                  num_heads,
                                  head_dim,
                                  eps,
                                  supported_attention_backends=supported_attention_backends,
                                  prefix=f"{prefix}.attn")

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = MLP(dim, int(dim * mlp_ratio), act_type="gelu_pytorch_tanh", prefix=f"{prefix}.ff")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = MLP(dim,
                              int(dim * context_mlp_ratio),
                              act_type="gelu_pytorch_tanh",
                              prefix=f"{prefix}.ff_context")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        (
            context_shift_msa,
            context_scale_msa,
            context_gate_msa,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.scale_shift_table_context.to(temb.dtype).chunk(6, dim=-1)
        norm_encoder_hidden_states = (self.norm1_context(encoder_hidden_states) *
                                      (1 + context_scale_msa) + context_shift_msa)

        attn_output, context_attn_output = self.attn(norm_hidden_states, norm_encoder_hidden_states, freqs_cis,
                                                     original_seq_len)
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * context_gate_msa

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (self.norm2_context(encoder_hidden_states) *
                                      (1 + context_scale_mlp) + context_shift_mlp)

        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.ff_context(norm_encoder_hidden_states) * context_gate_mlp
        return hidden_states, encoder_hidden_states


class PAN2Transformer3DModel(BaseDiT):
    """PAN2 DiT. The input stacks the noisy latents, the conditioning latents and a one-channel conditioning mask
    along channels, so one model serves text-to-video and first-frame image-to-video."""

    _fsdp_shard_conditions = PAN2VideoConfig()._fsdp_shard_conditions
    _compile_conditions = PAN2VideoConfig()._compile_conditions
    _supported_attention_backends = PAN2VideoConfig()._supported_attention_backends
    param_names_mapping = PAN2VideoConfig().param_names_mapping
    reverse_param_names_mapping = PAN2VideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = PAN2VideoConfig().lora_param_names_mapping

    def __init__(self, config: PAN2VideoConfig, hf_config: dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, hf_config=hf_config)
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = inner_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.out_channels = config.out_channels
        self.patch_size = tuple(config.patch_size)
        self.rope_axes_dim = tuple(config.rope_axes_dim)
        if sum(self.rope_axes_dim) != config.attention_head_dim:
            raise ValueError(f"rope_axes_dim {self.rope_axes_dim} must sum to the head dim {config.attention_head_dim}")
        if config.num_attention_heads % get_sp_world_size() != 0:
            raise ValueError(f"The number of attention heads ({config.num_attention_heads}) must be divisible by the "
                             f"sequence parallel size ({get_sp_world_size()})")

        self.x_embedder = PatchEmbed(self.patch_size,
                                     config.in_channels,
                                     inner_dim,
                                     prefix=f"{config.prefix}.x_embedder")
        self.time_embedder = TimestepEmbedder(inner_dim, act_layer="silu", frequency_embedding_size=256)

        self.context_norm = RMSNorm(config.text_embed_dim, eps=1e-5)
        self.context_proj = ReplicatedLinear(config.text_embed_dim,
                                             inner_dim,
                                             bias=True,
                                             prefix=f"{config.prefix}.context_proj")
        self.context_refiner = nn.ModuleList([
            PAN2TokenRefinerBlock(inner_dim,
                                  config.num_attention_heads,
                                  config.attention_head_dim,
                                  config.refiner_mlp_ratio,
                                  eps=1e-5,
                                  supported_attention_backends=self._supported_attention_backends,
                                  prefix=f"{config.prefix}.context_refiner.{i}")
            for i in range(config.num_refiner_layers)
        ])
        self.context_type_embedding = nn.Parameter(torch.zeros(inner_dim))

        self.transformer_blocks = nn.ModuleList([
            PAN2TransformerBlock(inner_dim,
                                 config.num_attention_heads,
                                 config.attention_head_dim,
                                 config.mlp_ratio,
                                 config.context_mlp_ratio,
                                 supported_attention_backends=self._supported_attention_backends,
                                 prefix=f"{config.prefix}.transformer_blocks.{i}") for i in range(config.num_layers)
        ])

        self.norm_out = PAN2AdaLayerNormOut(inner_dim, eps=1e-6, prefix=f"{config.prefix}.norm_out")
        self.proj_out = ReplicatedLinear(inner_dim,
                                         self.patch_size[0] * self.patch_size[1] * self.patch_size[2] *
                                         self.out_channels,
                                         bias=True,
                                         prefix=f"{config.prefix}.proj_out")

        self.gradient_checkpointing = False
        self.__post_init__()

    def _rotary_pos_embed(self, num_frames: int, height: int, width: int,
                          device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Half-dim cos/sin of the 3-axis ``(t, h, w)`` RoPE over the full (unsharded) video token grid."""
        axes = [torch.arange(size, device=device, dtype=torch.float32) for size in (num_frames, height, width)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        grid[0] = grid[0] * self.config.rope_temporal_scale
        cos_sin = [
            get_1d_rotary_pos_embed(dim, grid[i].reshape(-1), theta=self.config.rope_theta, use_real=False)
            for i, dim in enumerate(self.rope_axes_dim)
        ]
        freqs_cos = torch.cat([cos for cos, _ in cos_sin], dim=1).float()
        freqs_sin = torch.cat([sin for _, sin in cos_sin], dim=1).float()
        return freqs_cos, freqs_sin

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        freqs_cis = self._rotary_pos_embed(post_patch_num_frames, post_patch_height, post_patch_width,
                                           hidden_states.device)

        hidden_states = self.x_embedder(hidden_states)
        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)

        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        temb = self.time_embedder(timestep).to(hidden_states.dtype)

        encoder_hidden_states, _ = self.context_proj(self.context_norm(encoder_hidden_states))
        for refiner_block in self.context_refiner:
            encoder_hidden_states = refiner_block(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + self.context_type_embedding

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, freqs_cis, original_seq_len)
            else:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, freqs_cis,
                                                             original_seq_len)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states, _ = self.proj_out(hidden_states)

        # proj_out emits channels before the patch offsets: (C, p_t, p_h, p_w).
        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                                              -1, p_t, p_h, p_w)
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = PAN2Transformer3DModel
