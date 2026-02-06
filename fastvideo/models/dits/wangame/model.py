# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn

from fastvideo.configs.models.dits.wangamevideo import WanGameVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (_apply_rotary_emb,
                                               get_rotary_pos_embed)
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import WanI2VCrossAttention
from fastvideo.platforms import AttentionBackendEnum, current_platform

# Import ActionModule
from fastvideo.models.dits.wangame.hyworld_action_module import WanGameActionTimeTextImageEmbedding, WanGameActionSelfAttention

logger = init_logger(__name__)


class WanGameCrossAttention(WanI2VCrossAttention):
    def forward(self, x, context, context_lens=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).view(
            b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
        img_x = self.attn(q, k_img, v_img)

        # output
        x = img_x.flatten(2)
        x, _ = self.to_out(x)
        return x

class WanGameActionTransformerBlock(nn.Module):
    """
    Transformer block for WAN Action model with support for:
    - Self-attention with RoPE and camera PRoPE
    - Cross-attention with text/image context
    - Feed-forward network with AdaLN modulation
    """

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        
        self.attn1 = WanGameActionSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps)
        
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        dim_head = dim // num_heads

        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"QK Norm type {qk_norm} not supported")

        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            compute_dtype=torch.float32)

        # 2. Cross-attention (I2V only for now)
        self.attn2 = WanGameCrossAttention(dim,
                                          num_heads,
                                          qk_norm=qk_norm,
                                          eps=eps)
        # norm3 for FFN input 
        self.norm3 = LayerNormScaleShift(dim, norm_type="layer", eps=eps,
                                         elementwise_affine=False)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # PRoPE output projection (initialized via add_discrete_action_parameters on the model)
        self.to_out_prope = nn.ModuleList([
            nn.Linear(dim, dim, bias=True),
        ])
        nn.init.zeros_(self.to_out_prope[0].weight)
        if self.to_out_prope[0].bias is not None:
            nn.init.zeros_(self.to_out_prope[0].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        is_cache: bool = False,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)

        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # Cast temb to float32 for scale/shift computation
        e = self.scale_shift_table + temb.float()
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=2)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
            (1 + scale_msa) + shift_msa).to(orig_dtype).flatten(1, 2)
        
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

        # Self-attention with optional camera PRoPE
        attn_output_rope, attn_output_prope = self.attn1(
            query, key, value, freqs_cis,
            kv_cache, current_start, cache_start, viewmats, Ks,
            is_cache=is_cache
        )
        # Combine rope and prope outputs
        attn_output_rope = attn_output_rope.flatten(2)
        attn_output_rope, _ = self.to_out(attn_output_rope)
        attn_output_prope = attn_output_prope.flatten(2)
        attn_output_prope = self.to_out_prope[0](attn_output_prope)
        attn_output = attn_output_rope.squeeze(1) + attn_output_prope.squeeze(1)

        # Self-attention residual + norm in float32
        null_shift = null_scale = torch.zeros(1, device=hidden_states.device, dtype=torch.float32)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states.float(), attn_output.float(), gate_msa, null_shift, null_scale)
        hidden_states = hidden_states.type_as(attn_output)
        norm_hidden_states = norm_hidden_states.type_as(attn_output)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states.to(orig_dtype),
                                 context=encoder_hidden_states,
                                 context_lens=None)
        # Cross-attention residual in bfloat16
        hidden_states = hidden_states + attn_output
        
        # norm3 for FFN input in float32
        norm_hidden_states = self.norm3(
            hidden_states.float(), c_shift_msa, c_scale_msa
        ).type_as(hidden_states)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states.to(orig_dtype))
        hidden_states = self.mlp_residual(hidden_states.float(), ff_output.float(), c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)  # Cast back to original dtype

        return hidden_states

class WanGameActionTransformer3DModel(BaseDiT):
    """
    WAN Action Transformer 3D Model for video generation with action conditioning.
    
    Extends the base WAN video model with:
    - Action embedding support for controllable generation
    - camera PRoPE attention for 3D-aware generation
    - KV caching for autoregressive inference
    """
    _fsdp_shard_conditions = WanGameVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanGameVideoConfig()._compile_conditions
    _supported_attention_backends = WanGameVideoConfig()._supported_attention_backends
    param_names_mapping = WanGameVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanGameVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanGameVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanGameVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size
        self.inner_dim = inner_dim

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings (with action and text support)
        self.condition_embedder = WanGameActionTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            WanGameActionTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.local_attn_size,
                config.sink_size,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                config.added_kv_proj_dim,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.blocks.{i}")
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Causal-specific
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3

        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor],
        guidance=None,
        action: torch.Tensor | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        is_cache: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for both training and inference with KV caching.
        
        Args:
            hidden_states: Video latents [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Timestep tensor
            encoder_hidden_states_image: Optional image embeddings
            action: Action tensor [B, T] for per-frame conditioning
            viewmats: Camera view matrices for PRoPE [B, T, 4, 4]
            Ks: Camera intrinsics for PRoPE [B, T, 3, 3]
            kv_cache: KV cache for autoregressive inference (list of dicts per layer)
            crossattn_cache: Cross-attention cache for inference
            current_start: Current position for KV cache
            cache_start: Cache start position
            start_frame: RoPE offset for new frames in autoregressive mode
            is_cache: If True, populate KV cache and return early (cache-only mode)
        """
        orig_dtype = hidden_states.dtype
        # Handle encoder_hidden_states (text embeddings) - can be None, list, or tensor
        if encoder_hidden_states is None:
            pass  # Will be handled by condition_embedder (returns zeros)
        elif isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0] if len(encoder_hidden_states) > 0 else None
        # Handle encoder_hidden_states_image
        if isinstance(encoder_hidden_states_image, list) and len(encoder_hidden_states_image) > 0:
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
            (post_patch_num_frames * get_sp_world_size(), post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.dim() == 2:
            timestep = timestep.flatten()

        # Pad text embeddings to text_len if provided
        if encoder_hidden_states is not None and encoder_hidden_states.size(1) > 0:
            encoder_hidden_states = torch.cat([
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, self.text_len - encoder_hidden_states.size(1), encoder_hidden_states.size(2))
            ], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, action, encoder_hidden_states, encoder_hidden_states_image=encoder_hidden_states_image)
        
        # condition_embedder returns:
        # - temb: [B*T, dim] where T = post_patch_num_frames
        # - timestep_proj: [B*T, 6*dim]
        # Reshape to [B, T, 6, dim] for transformer blocks
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size))  # [B*T, 6, dim]
        timestep_proj = timestep_proj.view(batch_size, post_patch_num_frames, 6, self.hidden_size)  # [B, T, 6, dim]

        # Concatenate text and image embeddings for cross-attention
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        
        encoder_hidden_states = encoder_hidden_states.to(orig_dtype) if current_platform.is_mps() else encoder_hidden_states

        # Transformer blocks
        for block_idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, freqs_cis,
                    kv_cache[block_idx] if kv_cache else None,
                    crossattn_cache[block_idx] if crossattn_cache else None,
                    current_start, cache_start,
                    viewmats, Ks, is_cache)
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, freqs_cis,
                    kv_cache[block_idx] if kv_cache else None,
                    crossattn_cache[block_idx] if crossattn_cache else None,
                    current_start, cache_start,
                    viewmats, Ks, is_cache)

        # If cache-only mode, return early
        if is_cache:
            return kv_cache

        # Output norm, projection & unpatchify
        # temb is [B*T, dim], reshape to [B, T, 1, dim]
        temb = temb.view(batch_size, post_patch_num_frames, -1).unsqueeze(2)  # [B, T, 1, dim]
        
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, p_t, p_h, p_w,
                                              -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output