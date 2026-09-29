# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE dense transformer (Diffusers ``WanVACETransformer3DModel`` parity)."""

import math
from typing import Any

import torch
import torch.nn as nn

from fastvideo.attention.selector import get_env_variable_attn_backend
from fastvideo.distributed.communication_op import (sequence_model_parallel_all_gather_with_unpad,
                                                    sequence_model_parallel_shard)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import LayerNormScaleShift
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.wan.transformer import WanTimeTextImageEmbedding, WanTransformerBlock
from fastvideo.models.wan.vace_config import WanVACEVideoConfig
from fastvideo.platforms import AttentionBackendEnum, current_platform


class WanVACEMainBlock(WanTransformerBlock):
    """Wan main block with Diffusers VACE BF16 residual boundaries.

    Diffusers Wan-VACE rounds gated self-attention and FFN residuals in BF16
    before the following norm. The base :class:`WanTransformerBlock` keeps the
    historical fused residual helpers; this subclass overrides the three residual
    hooks so parity matches without patching instances at runtime.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str,
        cross_attn_norm: bool,
        eps: float,
        added_kv_proj_dim: int | None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim,
                         ffn_dim,
                         num_heads,
                         qk_norm,
                         cross_attn_norm,
                         eps,
                         added_kv_proj_dim,
                         supported_attention_backends,
                         quant_config=quant_config,
                         prefix=prefix)
        inner_dim = self.hidden_dim
        self.norm_q = nn.RMSNorm(inner_dim, eps=eps)
        self.norm_k = nn.RMSNorm(inner_dim, eps=eps)
        self.attn2.norm_q = nn.RMSNorm(inner_dim, eps=eps)
        self.attn2.norm_k = nn.RMSNorm(inner_dim, eps=eps)
        if added_kv_proj_dim is not None:
            self.attn2.norm_added_k = nn.RMSNorm(inner_dim, eps=eps)

    def _self_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
        gate_msa: torch.Tensor,
        orig_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = (hidden_states.float() + attn_output * gate_msa).to(orig_dtype)
        norm_hidden_states = self.self_attn_residual_norm.norm(hidden_states.float()).to(orig_dtype)
        return hidden_states, norm_hidden_states

    def _cross_attn_residual(
        self,
        hidden_states: torch.Tensor,
        norm_hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
        c_shift_msa: torch.Tensor,
        c_scale_msa: torch.Tensor,
        orig_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states + attn_output
        norm_hidden_states = (self.cross_attn_residual_norm.norm(hidden_states.float()) * (1 + c_scale_msa)
                              + c_shift_msa).to(orig_dtype)
        return hidden_states, norm_hidden_states

    def _ffn_residual(
        self,
        hidden_states: torch.Tensor,
        ff_output: torch.Tensor,
        c_gate_msa: torch.Tensor,
        orig_dtype: torch.dtype,
    ) -> torch.Tensor:
        return (hidden_states.float() + ff_output.float() * c_gate_msa).to(orig_dtype)


class WanVACETransformerBlock(WanVACEMainBlock):
    """VACE control block: a :class:`WanVACEMainBlock` over the control stream.

    Block 0 projects the control tokens and adds the main hidden states first;
    every block emits a projected hint that is added to its paired main block.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str,
        cross_attn_norm: bool,
        eps: float,
        added_kv_proj_dim: int | None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None,
        apply_input_projection: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim,
                         ffn_dim,
                         num_heads,
                         qk_norm,
                         cross_attn_norm,
                         eps,
                         added_kv_proj_dim,
                         supported_attention_backends,
                         quant_config=quant_config,
                         prefix=prefix)
        self.proj_in = (ReplicatedLinear(dim, dim, quant_config=quant_config, prefix=f"{prefix}.proj_in")
                        if apply_input_projection else None)
        self.proj_out = ReplicatedLinear(dim, dim, quant_config=quant_config, prefix=f"{prefix}.proj_out")

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proj_in is not None:
            control_hidden_states, _ = self.proj_in(control_hidden_states.contiguous())
            control_hidden_states = control_hidden_states + hidden_states
        control_hidden_states = super().forward(control_hidden_states, encoder_hidden_states, temb, freqs_cis,
                                                original_seq_len)
        hint, _ = self.proj_out(control_hidden_states)
        return hint, control_hidden_states


class WanVACETransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = WanVACEVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVACEVideoConfig()._compile_conditions
    _supported_attention_backends = WanVACEVideoConfig()._supported_attention_backends
    param_names_mapping = WanVACEVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVACEVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVACEVideoConfig().lora_param_names_mapping

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        # Diffusers keeps modulation tables, time embeddings, and norm2 in FP32.
        if (name == "scale_shift_table" or name.endswith(".scale_shift_table")
                or name.startswith("condition_embedder.time_embedder.")
                or ".self_attn_residual_norm.norm." in name):
            return torch.float32
        return default_dtype

    def __init__(self, config: WanVACEVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        supported_backends = config._supported_attention_backends
        requested_backend = config._resolved_attention_backend or get_env_variable_attn_backend()
        if requested_backend is not None and requested_backend not in supported_backends:
            raise ValueError(f"Wan-VACE requires dense attention; requested {requested_backend.name}. "
                             "Choose TORCH_SDPA, FLASH_ATTN, or another supported dense backend.")
        self.quant_config = config.quant_config
        arch = config.arch_config

        inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        self.num_channels_latents = arch.num_channels_latents
        self.patch_size = arch.patch_size
        self.text_len = arch.text_len
        self.vace_layers = tuple(arch.vace_layers)

        assert arch.num_attention_heads % get_sp_world_size() == 0, (
            f"num_attention_heads ({arch.num_attention_heads}) must be divisible by SP size "
            f"({get_sp_world_size()})")

        self.patch_embedding = PatchEmbed(in_chans=arch.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=arch.patch_size,
                                          flatten=False)
        self.vace_patch_embedding = PatchEmbed(in_chans=arch.vace_in_channels,
                                               embed_dim=inner_dim,
                                               patch_size=arch.patch_size,
                                               flatten=False)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=arch.freq_dim,
            text_embed_dim=arch.text_dim,
            image_embed_dim=arch.image_dim,
            cast_temb_to_context_dtype=True,
        )

        self.blocks = nn.ModuleList([
            WanVACEMainBlock(inner_dim,
                             arch.ffn_dim,
                             arch.num_attention_heads,
                             arch.qk_norm,
                             arch.cross_attn_norm,
                             arch.eps,
                             arch.added_kv_proj_dim,
                             supported_backends,
                             quant_config=config.quant_config,
                             prefix=f"{config.prefix}.blocks.{i}") for i in range(arch.num_layers)
        ])
        self.vace_blocks = nn.ModuleList([
            WanVACETransformerBlock(inner_dim,
                                    arch.ffn_dim,
                                    arch.num_attention_heads,
                                    arch.qk_norm,
                                    arch.cross_attn_norm,
                                    arch.eps,
                                    arch.added_kv_proj_dim,
                                    supported_backends,
                                    apply_input_projection=(i == 0),
                                    quant_config=config.quant_config,
                                    prefix=f"{config.prefix}.vace_blocks.{i}")
            for i in range(len(arch.vace_layers))
        ])

        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=arch.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32,
                                            compute_dtype=torch.float32)
        self.proj_out = nn.Linear(inner_dim, arch.out_channels * math.prod(arch.patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.__post_init__()

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
                control_hidden_states: torch.Tensor | None = None,
                control_hidden_states_scale: torch.Tensor | list[float] | float | None = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        if control_hidden_states is None:
            raise ValueError("WanVACETransformer3DModel requires control_hidden_states.")
        if encoder_hidden_states is not None and not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = (encoder_hidden_states_image[0]
                                           if len(encoder_hidden_states_image) > 0 else None)

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed((post_patch_num_frames, post_patch_height, post_patch_width),
                                                    self.hidden_size,
                                                    self.num_attention_heads,
                                                    rope_dim_list,
                                                    dtype=torch.float32 if current_platform.is_mps() else torch.float64,
                                                    rope_theta=10000)
        freqs_cis = (freqs_cos.to(hidden_states.device).float(), freqs_sin.to(hidden_states.device).float())

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        control_hidden_states = self.vace_patch_embedding(control_hidden_states)
        control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
        if control_hidden_states.shape[1] < hidden_states.shape[1]:
            padding = control_hidden_states.new_zeros(batch_size,
                                                      hidden_states.shape[1] - control_hidden_states.shape[1],
                                                      control_hidden_states.shape[2])
            control_hidden_states = torch.cat([control_hidden_states, padding], dim=1)

        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
        control_hidden_states, _ = sequence_model_parallel_shard(control_hidden_states, dim=1)

        if control_hidden_states_scale is None:
            control_hidden_states_scale = 1.0
        if isinstance(control_hidden_states_scale, bool):
            raise TypeError("control_hidden_states_scale must be a float, list of floats, or tensor.")
        control_hidden_states_scale = torch.as_tensor(control_hidden_states_scale,
                                                      device=hidden_states.device,
                                                      dtype=hidden_states.dtype).reshape(-1)
        if control_hidden_states_scale.numel() == 1:
            control_hidden_states_scale = control_hidden_states_scale.expand(len(self.vace_layers))
        if control_hidden_states_scale.numel() != len(self.vace_layers):
            raise ValueError(f"control_hidden_states_scale length {control_hidden_states_scale.numel()} "
                             f"!= vace_layers {len(self.vace_layers)}")

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        control_hints: dict[int, torch.Tensor] = {}
        for layer_idx, block, scale in zip(self.vace_layers, self.vace_blocks, control_hidden_states_scale.unbind()):
            hint, control_hidden_states = block(hidden_states, encoder_hidden_states, control_hidden_states,
                                                timestep_proj, freqs_cis, original_seq_len)
            control_hints[layer_idx] = hint * scale

        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, original_seq_len)
            if layer_idx in control_hints:
                hidden_states = hidden_states + control_hints[layer_idx]

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                                              p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = WanVACETransformer3DModel
