# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import fastvideo.v1.envs as envs
from fastvideo.v1.attention import (DistributedAttention,
                                    DistributedAttention_VSA, LocalAttention)
from fastvideo.v1.configs.models.dits import WanVideoConfig
from fastvideo.v1.configs.sample.wan import WanTeaCacheParams
from fastvideo.v1.distributed.parallel_state import get_sp_world_size
from fastvideo.v1.forward_context import get_forward_context
from fastvideo.v1.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                           RMSNorm, ScaleResidual,
                                           ScaleResidualLayerNormScaleShift)
from fastvideo.v1.layers.linear import ReplicatedLinear
# from torch.nn import RMSNorm
# TODO: RMSNorm ....
from fastvideo.v1.layers.mlp import MLP
from fastvideo.v1.layers.rotary_embedding import (_apply_rotary_emb,
                                                  get_rotary_pos_embed)
from fastvideo.v1.layers.visual_embedding import (ModulateProjection,
                                                  PatchEmbed, TimestepEmbedder)
from fastvideo.v1.models.dits.base import CachableDiT
from fastvideo.v1.platforms import AttentionBackendEnum


class WanImageEmbedding(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self,
                encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        dtype = encoder_hidden_states_image.dtype
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states).to(dtype)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu")
        self.time_modulation = ModulateProjection(dim,
                                                  factor=6,
                                                  act_layer="silu")
        self.text_embedder = MLP(text_embed_dim,
                                 dim,
                                 dim,
                                 bias=True,
                                 act_type="gelu_pytorch_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_modulation(temb)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 parallel_attention=False) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention

        # layers
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_lens: int):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: Optional[Tuple[AttentionBackendEnum,
                                                     ...]] = None
    ) -> None:
        super().__init__(dim, num_heads, window_size, qk_norm, eps,
                         supported_attention_backends)

        self.add_k_proj = ReplicatedLinear(dim, dim)
        self.add_v_proj = ReplicatedLinear(dim, dim)
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_added_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)
        k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).view(
            b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
        img_x = self.attn(q, k_img, v_img)
        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


class WanTransformerBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            ffn_dim: int,
            num_heads: int,
            qk_norm: str = "rms_norm_across_heads",
            cross_attn_norm: bool = False,
            eps: float = 1e-6,
            added_kv_proj_dim: Optional[int] = None,
            supported_attention_backends: Optional[Tuple[AttentionBackendEnum,
                                                         ...]] = None,
            prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = DistributedAttention(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1")
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1)
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        # Apply rotary embeddings
        cos, sin = freqs_cis
        query, key = _apply_rotary_emb(query, cos, sin,
                                       is_neox_style=False), _apply_rotary_emb(
                                           key, cos, sin, is_neox_style=False)

        attn_output, _ = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformerBlock_VSA(nn.Module):

    def __init__(
            self,
            dim: int,
            ffn_dim: int,
            num_heads: int,
            qk_norm: str = "rms_norm_across_heads",
            cross_attn_norm: bool = False,
            eps: float = 1e-6,
            added_kv_proj_dim: Optional[int] = None,
            supported_attention_backends: Optional[Tuple[AttentionBackendEnum,
                                                         ...]] = None,
            prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_gate_compress = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = DistributedAttention_VSA(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1")
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1)
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        gate_compress, _ = self.to_gate_compress(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        gate_compress = gate_compress.squeeze(1).unflatten(
            2, (self.num_attention_heads, -1))

        # Apply rotary embeddings
        cos, sin = freqs_cis
        query, key = _apply_rotary_emb(query, cos, sin,
                                       is_neox_style=False), _apply_rotary_emb(
                                           key, cos, sin, is_neox_style=False)

        attn_output, _ = self.attn1(query,
                                    key,
                                    value,
                                    gate_compress=gate_compress)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformer3DModel(CachableDiT):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig(
    )._supported_attention_backends
    _param_names_mapping = WanVideoConfig()._param_names_mapping
    _reverse_param_names_mapping = WanVideoConfig()._reverse_param_names_mapping
    _lora_param_names_mapping = WanVideoConfig()._lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str,
                                                               Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        attn_backend = envs.FASTVIDEO_ATTENTION_BACKEND
        transformer_block = WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN" else WanTransformerBlock
        self.blocks = nn.ModuleList([
            transformer_block(inner_dim,
                              config.ffn_dim,
                              config.num_attention_heads,
                              config.qk_norm,
                              config.cross_attn_norm,
                              config.eps,
                              config.added_kv_proj_dim,
                              self._supported_attention_backends,
                              prefix=f"{config.prefix}.blocks.{i}")
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32,
                                            compute_dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # For type checking
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.is_even = True
        self.should_calc_even = True
        self.should_calc_odd = True
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.cnt = 0
        self.__post_init__()

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: Optional[Union[
                    torch.Tensor, List[torch.Tensor]]] = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        enable_teacache = forward_batch is not None and forward_batch.enable_teacache

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sp_world_size(), post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float64,
            rope_theta=10000)
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos.float(),
                     freqs_sin.float()) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        # if caching is enabled, we might be able to skip the forward pass
        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb)

        if should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            # if teacache is enabled, we need to cache the original hidden states
            if enable_teacache:
                original_hidden_states = hidden_states.clone()

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for block in self.blocks:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states,
                        timestep_proj, freqs_cis)
            else:
                for block in self.blocks:
                    hidden_states = block(hidden_states, encoder_hidden_states,
                                          timestep_proj, freqs_cis)

            # if teacache is enabled, we need to cache the original hidden states
            if enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2,
                                                                          dim=1)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, p_t, p_h, p_w,
                                              -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output

    def maybe_cache_states(self, hidden_states: torch.Tensor,
                           original_hidden_states: torch.Tensor) -> None:
        if self.is_even:
            self.previous_residual_even = hidden_states.squeeze(
                0) - original_hidden_states
        else:
            self.previous_residual_odd = hidden_states.squeeze(
                0) - original_hidden_states

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None or not forward_batch.enable_teacache:
            return False
        teacache_params = forward_batch.teacache_params
        assert teacache_params is not None, "teacache_params is not initialized"
        assert isinstance(
            teacache_params,
            WanTeaCacheParams), "teacache_params is not a WanTeaCacheParams"
        current_timestep = forward_context.current_timestep
        num_inference_steps = forward_batch.num_inference_steps

        # initialize the coefficients, cutoff_steps, and ret_steps
        coefficients = teacache_params.coefficients
        use_ret_steps = teacache_params.use_ret_steps
        cutoff_steps = teacache_params.get_cutoff_steps(num_inference_steps)
        ret_steps = teacache_params.ret_steps
        teacache_thresh = teacache_params.teacache_thresh

        if current_timestep == 0:
            self.cnt = 0

        timestep_proj = kwargs["timestep_proj"]
        temb = kwargs["temb"]
        modulated_inp = timestep_proj if use_ret_steps else temb

        if self.cnt % 2 == 0:  # even -> condition
            self.is_even = True
            if self.cnt < ret_steps or self.cnt >= cutoff_steps:
                self.should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                assert self.previous_e0_even is not None, "previous_e0_even is not initialized"
                assert self.accumulated_rel_l1_distance_even is not None, "accumulated_rel_l1_distance_even is not initialized"
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.previous_e0_even).abs().mean() /
                     self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < teacache_thresh:
                    self.should_calc_even = False
                else:
                    self.should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else:  # odd -> unconditon
            self.is_even = False
            if self.cnt < ret_steps or self.cnt >= cutoff_steps:
                self.should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                assert self.previous_e0_odd is not None, "previous_e0_odd is not initialized"
                assert self.accumulated_rel_l1_distance_odd is not None, "accumulated_rel_l1_distance_odd is not initialized"
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(
                    ((modulated_inp - self.previous_e0_odd).abs().mean() /
                     self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < teacache_thresh:
                    self.should_calc_odd = False
                else:
                    self.should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()
        self.cnt += 1
        should_skip_forward = False
        if self.is_even:
            if not self.should_calc_even:
                should_skip_forward = True
        else:
            if not self.should_calc_odd:
                should_skip_forward = True

        return should_skip_forward

    def retrieve_cached_states(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        if self.is_even:
            return hidden_states + self.previous_residual_even
        else:
            return hidden_states + self.previous_residual_odd
