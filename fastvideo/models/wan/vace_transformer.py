# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE dense transformer (Diffusers ``WanVACETransformer3DModel`` parity)."""

import math
from typing import Any

import torch
import torch.nn as nn

import fastvideo.envs as envs
from fastvideo.attention import DistributedAttention
from fastvideo.distributed.communication_op import (sequence_model_parallel_all_gather_with_unpad,
                                                    sequence_model_parallel_shard)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift, RMSNorm,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.wan.transformer import (WanI2VCrossAttention, WanT2VCrossAttention,
                                              WanTimeTextImageEmbedding, WanTransformerBlock,
                                              WanTransformerBlock_VSA)
from fastvideo.models.wan.vace_config import WanVACEVideoConfig
from fastvideo.platforms import AttentionBackendEnum, current_platform

logger = init_logger(__name__)


class WanVACETransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 apply_input_projection: bool = False,
                 apply_output_projection: bool = False,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.proj_in = (ReplicatedLinear(dim, dim, prefix=f"{prefix}.proj_in")
                        if apply_input_projection else None)
        self.proj_out = (ReplicatedLinear(dim, dim, prefix=f"{prefix}.proj_out")
                         if apply_output_projection else None)

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True, prefix=f"{prefix}.to_q")
        self.to_k = ReplicatedLinear(dim, dim, bias=True, prefix=f"{prefix}.to_k")
        self.to_v = ReplicatedLinear(dim, dim, bias=True, prefix=f"{prefix}.to_v")
        self.to_out = ReplicatedLinear(dim, dim, bias=True, prefix=f"{prefix}.to_out")
        self.attn1 = DistributedAttention(num_heads=num_heads,
                                          head_size=dim // num_heads,
                                          causal=False,
                                          supported_attention_backends=supported_attention_backends,
                                          prefix=f"{prefix}.attn1")
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unsupported qk_norm for VACE: {qk_norm!r}")

        self.attn2 = (WanI2VCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps, prefix=f"{prefix}.attn2")
                      if added_kv_proj_dim is not None else WanT2VCrossAttention(dim,
                                                                                num_heads,
                                                                                qk_norm=qk_norm,
                                                                                eps=eps,
                                                                                prefix=f"{prefix}.attn2"))
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(dim,
                                                                      norm_type="layer",
                                                                      eps=eps,
                                                                      elementwise_affine=True,
                                                                      dtype=torch.float32,
                                                                      compute_dtype=torch.float32)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(dim,
                                                                         norm_type="layer",
                                                                         eps=eps,
                                                                         elementwise_affine=False,
                                                                         dtype=torch.float32,
                                                                         compute_dtype=torch.float32)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh", prefix=f"{prefix}.ffn")
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.proj_in is not None:
            control_hidden_states, _ = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()).chunk(6, dim=1)
        orig_dtype = control_hidden_states.dtype

        norm_hidden_states = (self.norm1(control_hidden_states.float()) * (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        attn_output, _ = self.attn1(query, key, value, original_seq_len, freqs_cis=freqs_cis)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)
        null_shift = null_scale = torch.tensor([0], device=control_hidden_states.device)
        norm_hidden_states, control_hidden_states = self.self_attn_residual_norm(
            control_hidden_states, attn_output, gate_msa, null_shift, null_scale)
        control_hidden_states = control_hidden_states.to(orig_dtype)

        attn_output = self.attn2(norm_hidden_states, context=encoder_hidden_states, context_lens=None)
        norm_hidden_states, control_hidden_states = self.cross_attn_residual_norm(
            control_hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)
        control_hidden_states = control_hidden_states.to(orig_dtype)

        norm_hidden_states = (self.norm3(control_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).to(
            orig_dtype)
        ff_output = self.ffn(norm_hidden_states)
        control_hidden_states = (control_hidden_states.float() + ff_output.float() * c_gate_msa).to(orig_dtype)

        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states, _ = self.proj_out(control_hidden_states)
        return conditioning_states, control_hidden_states


class WanVACETransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = WanVACEVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVACEVideoConfig()._compile_conditions
    _supported_attention_backends = WanVACEVideoConfig()._supported_attention_backends
    param_names_mapping = WanVACEVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVACEVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVACEVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVACEVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
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
        )

        attn_backend = envs.FASTVIDEO_ATTENTION_BACKEND
        transformer_block = (WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN" else WanTransformerBlock)
        self.blocks = nn.ModuleList([
            transformer_block(inner_dim,
                              arch.ffn_dim,
                              arch.num_attention_heads,
                              arch.qk_norm,
                              arch.cross_attn_norm,
                              arch.eps,
                              arch.added_kv_proj_dim,
                              self._supported_attention_backends,
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
                                    apply_input_projection=(i == 0),
                                    apply_output_projection=True,
                                    supported_attention_backends=self._supported_attention_backends,
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
        self.gradient_checkpointing = False
        self.__post_init__()

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
                control_hidden_states: torch.Tensor | None = None,
                control_hidden_states_scale: torch.Tensor | list[float] | float | None = None,
                guidance=None,
                r_timestep: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        if control_hidden_states is None:
            raise ValueError("WanVACETransformer3DModel requires control_hidden_states.")
        orig_dtype = hidden_states.dtype
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
            control_hidden_states_scale = [1.0] * len(self.vace_layers)
        if isinstance(control_hidden_states_scale, (int, float)):
            control_hidden_states_scale = [float(control_hidden_states_scale)] * len(self.vace_layers)
        if isinstance(control_hidden_states_scale, torch.Tensor):
            control_hidden_states_scale = control_hidden_states_scale.tolist()
        if len(control_hidden_states_scale) != len(self.vace_layers):
            raise ValueError(f"control_hidden_states_scale length {len(control_hidden_states_scale)} "
                             f"!= vace_layers {len(self.vace_layers)}")

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        control_hidden_states_list: list[tuple[torch.Tensor | None, float]] = []
        for block, scale in zip(self.vace_blocks, control_hidden_states_scale):
            conditioning_states, control_hidden_states = block(hidden_states,
                                                               encoder_hidden_states,
                                                               control_hidden_states,
                                                               timestep_proj,
                                                               freqs_cis,
                                                               original_seq_len)
            control_hidden_states_list.append((conditioning_states, scale))
        control_hidden_states_list = control_hidden_states_list[::-1]

        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, original_seq_len)
            if layer_idx in self.vace_layers:
                control_hint, scale = control_hidden_states_list.pop()
                if control_hint is not None:
                    hidden_states = hidden_states + control_hint.to(hidden_states.device) * scale

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                                              p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = WanVACETransformer3DModel
