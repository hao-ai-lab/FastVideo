# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models.dits.lingbotworld import LingBotWorldVideoConfig
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard)
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
# from torch.nn import RMSNorm
# TODO: RMSNorm ....
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import (PatchEmbed, WanCamControlPatchEmbedding)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import (
    WanI2VCrossAttention,
    WanT2VCrossAttention,
    WanTimeTextImageEmbedding,
)
from fastvideo.platforms import AttentionBackendEnum, current_platform

from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.distributed.utils import create_attention_mask_for_padding

logger = init_logger(__name__)


class LingBotWorldCamConditioner(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cam_injector = MLP(dim, dim, dim, bias=True, act_type="silu")
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        c2ws_plucker_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        if c2ws_plucker_emb is None:
            return hidden_states
        assert c2ws_plucker_emb.shape == hidden_states.shape, (
            f"c2ws_plucker_emb shape must match hidden_states shape, got "
            f"{tuple(c2ws_plucker_emb.shape)} vs {tuple(hidden_states.shape)}"
        )
        c2ws_hidden_states = self.cam_injector(c2ws_plucker_emb)
        c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
        cam_scale = self.cam_scale_layer(c2ws_hidden_states)
        cam_shift = self.cam_shift_layer(c2ws_hidden_states)
        return (1.0 + cam_scale) * hidden_states + cam_shift


class LingBotWorldTransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
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
            raise NotImplementedError(
                f"QK Norm type '{qk_norm}' not supported")
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
        self.cam_conditioner = LingBotWorldCamConditioner(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        c2ws_plucker_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32

        if temb.dim() == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
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
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output, _ = self.attn1(query, key, value, freqs_cis=freqs_cis, attention_mask=attention_mask)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)
        # Inject camera condition
        # must be applied after the self-attention residual update.
        hidden_states = self.cam_conditioner(hidden_states, c2ws_plucker_emb)
        norm_hidden_states = self.self_attn_residual_norm.norm(hidden_states)
        norm_hidden_states = norm_hidden_states.to(orig_dtype)

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


class LingBotWorldTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = LingBotWorldVideoConfig()._fsdp_shard_conditions
    _compile_conditions = LingBotWorldVideoConfig()._compile_conditions
    _supported_attention_backends = LingBotWorldVideoConfig(
    )._supported_attention_backends
    param_names_mapping = LingBotWorldVideoConfig().param_names_mapping
    reverse_param_names_mapping = LingBotWorldVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LingBotWorldVideoConfig().lora_param_names_mapping

    def __init__(self, config: LingBotWorldVideoConfig, hf_config: dict[str,
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

        assert config.num_attention_heads % get_sp_world_size() == 0, f"The number of attention heads ({config.num_attention_heads}) must be divisible by the sequence parallel size ({get_sp_world_size()})"

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)
        self.patch_embedding_wancamctrl = WanCamControlPatchEmbedding(in_chans=6 * 64,
                                                                      embed_dim=inner_dim,
                                                                      patch_size=config.patch_size)
        self.c2ws_mlp = MLP(inner_dim, inner_dim, inner_dim, bias=True, act_type="silu")

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        transformer_block = LingBotWorldTransformerBlock
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
        self._logged_attention_mask = False
        self.__post_init__()

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                guidance=None,
                c2ws_plucker_emb: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if encoder_hidden_states is not None and not isinstance(encoder_hidden_states, torch.Tensor):
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
            (post_patch_num_frames, post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000)
        freqs_cis = (freqs_cos.to(hidden_states.device).float(),
                     freqs_sin.to(hidden_states.device).float())

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        c2ws_hidden_states = None
        if c2ws_plucker_emb is not None:
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(
                c2ws_plucker_emb.to(device=hidden_states.device, dtype=hidden_states.dtype)
            )
            c2ws_hidden_states = self.c2ws_mlp(c2ws_plucker_emb)
            c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states

        # Shard with padding support - returns (sharded_tensor, original_seq_len)
        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
        
        # Shard c2ws_plucker_emb
        if c2ws_plucker_emb is not None:
            c2ws_plucker_emb, _ = sequence_model_parallel_shard(c2ws_plucker_emb, dim=1)
        
        # Create attention mask for padded tokens if padding was applied
        current_seq_len = hidden_states.shape[1]
        sp_world_size = get_sp_world_size()
        padded_seq_len = current_seq_len * sp_world_size
        
        if padded_seq_len > original_seq_len:
            if not self._logged_attention_mask:
                logger.info(f"Padding applied, original seq len: {original_seq_len}, padded seq len: {padded_seq_len}")
                self._logged_attention_mask = True
            attention_mask = create_attention_mask_for_padding(
                seq_len=original_seq_len,
                padded_seq_len=padded_seq_len,
                batch_size=batch_size,
                device=hidden_states.device,
            )
        else:
            if not self._logged_attention_mask:
                logger.info(f"Padding not applied")
                self._logged_attention_mask = True
            attention_mask = None

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len)
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states_image

        if current_platform.is_mps() or current_platform.is_npu():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)
        else:
            encoder_hidden_states = encoder_hidden_states # cast to orig_dtype for MPS & NPU

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis, attention_mask, c2ws_plucker_emb)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                      timestep_proj, freqs_cis, attention_mask,
                                      c2ws_plucker_emb)
        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            

        hidden_states = self.norm_out(hidden_states, shift, scale)

        # Gather and unpad in one operation
        hidden_states = sequence_model_parallel_all_gather_with_unpad(
            hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, p_t, p_h, p_w,
                                              -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output

# Entry point for model registry
EntryClass = LingBotWorldTransformer3DModel
