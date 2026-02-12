# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
import torch.distributed as dist

from fastvideo.attention import LocalAttention
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
from fastvideo.models.dits.wangame.hyworld_action_module import (
    WanGameActionTimeImageEmbedding,
    WanGameActionSelfAttention
)
from fastvideo.models.dits.hyworld.camera_rope import prope_qkv

logger = init_logger(__name__)


class CausalWanGameCrossAttention(WanI2VCrossAttention):
    """Cross-attention for WanGame causal model"""

    def forward(self, x, context, context_lens=None, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache: Optional cache dict for inference
        """
        context_img = context
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).view(
                    b, -1, n, d)
                v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
                crossattn_cache["k"] = k_img
                crossattn_cache["v"] = v_img
            else:
                k_img = crossattn_cache["k"]
                v_img = crossattn_cache["v"]
        else:
            k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).view(
                b, -1, n, d)
            v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)

        img_x = self.attn(q, k_img, v_img)

        # output
        x = img_x.flatten(2)
        x, _ = self.to_out(x)
        return x


class CausalWanGameActionSelfAttention(WanGameActionSelfAttention):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm=True,
                 eps=1e-6) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # Local attention for KV-cache inference
        self.local_attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    @staticmethod
    def _masked_flex_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        padded_length = math.ceil(query.shape[1] / 128) * 128 - query.shape[1]
        if padded_length > 0:
            query = torch.cat(
                [
                    query,
                    torch.zeros(
                        [query.shape[0], padded_length, query.shape[2], query.shape[3]],
                        device=query.device,
                        dtype=value.dtype,
                    ),
                ],
                dim=1,
            )
            key = torch.cat(
                [
                    key,
                    torch.zeros(
                        [key.shape[0], padded_length, key.shape[2], key.shape[3]],
                        device=key.device,
                        dtype=value.dtype,
                    ),
                ],
                dim=1,
            )
            value = torch.cat(
                [
                    value,
                    torch.zeros(
                        [value.shape[0], padded_length, value.shape[2], value.shape[3]],
                        device=value.device,
                        dtype=value.dtype,
                    ),
                ],
                dim=1,
            )

        out = flex_attention(
            query=query.transpose(2, 1),
            key=key.transpose(2, 1),
            value=value.transpose(2, 1),
            block_mask=block_mask,
        ).transpose(2, 1)

        if padded_length > 0:
            out = out[:, :-padded_length]
        return out

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor],
                block_mask: BlockMask | None = None,
                kv_cache: dict | None = None,
                current_start: int = 0,
                cache_start: int | None = None,
                viewmats: torch.Tensor | None = None,
                Ks: torch.Tensor | None = None,
                is_cache: bool = False):
        """
        Forward pass with causal attention.
        """
        if cache_start is None:
            cache_start = current_start

        if kv_cache is None:
            if block_mask is None:
                raise ValueError(
                    "block_mask must be provided for causal training attention")
            if viewmats is None or Ks is None:
                raise ValueError(
                    "viewmats and Ks must be provided for WanGame causal attention")

            cos, sin = freqs_cis
            query_rope = _apply_rotary_emb(
                q, cos, sin, is_neox_style=False).type_as(v)
            key_rope = _apply_rotary_emb(
                k, cos, sin, is_neox_style=False).type_as(v)
            rope_output = self._masked_flex_attn(
                query_rope, key_rope, v, block_mask)

            # PRoPE path with the same causal mask.
            query_prope, key_prope, value_prope, apply_fn_o = prope_qkv(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                viewmats=viewmats,
                Ks=Ks,
                patches_x=40,
                patches_y=22,
            )
            query_prope = query_prope.transpose(1, 2)
            key_prope = key_prope.transpose(1, 2)
            value_prope = value_prope.transpose(1, 2)
            prope_output = self._masked_flex_attn(
                query_prope, key_prope, value_prope, block_mask)
            prope_output = apply_fn_o(
                prope_output.transpose(1, 2)).transpose(1, 2)

            return rope_output, prope_output
        else:
            # Inference mode with KV cache
            if viewmats is None or Ks is None:
                raise ValueError(
                    "viewmats and Ks must be provided for WanGame causal attention")

            cos, sin = freqs_cis
            roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
            roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)
            query_prope, key_prope, value_prope, apply_fn_o = prope_qkv(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                viewmats=viewmats,
                Ks=Ks,
                patches_x=40,
                patches_y=22,
            )
            query_prope = query_prope.transpose(1, 2).type_as(v)
            key_prope = key_prope.transpose(1, 2).type_as(v)
            value_prope = value_prope.transpose(1, 2).type_as(v)

            frame_seqlen = q.shape[1]
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            # rope+prope
            if kv_cache["k"].shape[-1] == self.head_dim:
                kv_cache["k"] = torch.cat(
                    [kv_cache["k"], torch.zeros_like(kv_cache["k"])], dim=-1)
                kv_cache["v"] = torch.cat(
                    [kv_cache["v"], torch.zeros_like(kv_cache["v"])], dim=-1)
            elif kv_cache["k"].shape[-1] != self.head_dim * 2:
                raise ValueError(
                    f"Unexpected kv_cache head dim: {kv_cache['k'].shape[-1]}, "
                    f"expected {self.head_dim} or {self.head_dim * 2}")

            cache_k_rope = kv_cache["k"][..., :self.head_dim]
            cache_k_prope = kv_cache["k"][..., self.head_dim:]
            cache_v_rope = kv_cache["v"][..., :self.head_dim]
            cache_v_prope = kv_cache["v"][..., self.head_dim:]

            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                cache_k_rope[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    cache_k_rope[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                cache_v_rope[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    cache_v_rope[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                cache_k_prope[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    cache_k_prope[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                cache_v_prope[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    cache_v_prope[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                cache_k_rope[:, local_start_index:local_end_index] = roped_key
                cache_v_rope[:, local_start_index:local_end_index] = v
                cache_k_prope[:, local_start_index:local_end_index] = key_prope
                cache_v_prope[:, local_start_index:local_end_index] = value_prope
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"] = kv_cache["k"].detach()
                kv_cache["v"] = kv_cache["v"].detach()
                cache_k_rope = kv_cache["k"][..., :self.head_dim]
                cache_k_prope = kv_cache["k"][..., self.head_dim:]
                cache_v_rope = kv_cache["v"][..., :self.head_dim]
                cache_v_prope = kv_cache["v"][..., self.head_dim:]
                # logger.info("kv_cache['k'] is in comp graph: %s", kv_cache["k"].requires_grad or kv_cache["k"].grad_fn is not None)
                cache_k_rope[:, local_start_index:local_end_index] = roped_key
                cache_v_rope[:, local_start_index:local_end_index] = v
                cache_k_prope[:, local_start_index:local_end_index] = key_prope
                cache_v_prope[:, local_start_index:local_end_index] = value_prope

            rope_x = self.local_attn(
                roped_query,
                cache_k_rope[:, max(0, local_end_index - self.max_attention_size):local_end_index],
                cache_v_rope[:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            prope_x = self.local_attn(
                query_prope,
                cache_k_prope[:, max(0, local_end_index - self.max_attention_size):local_end_index],
                cache_v_prope[:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            prope_x = apply_fn_o(prope_x.transpose(1, 2)).transpose(1, 2)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

            return rope_x, prope_x


class CausalWanGameActionTransformerBlock(nn.Module):

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

        self.attn1 = CausalWanGameActionSelfAttention(
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

        # 2. Cross-attention (I2V only)
        self.attn2 = CausalWanGameCrossAttention(dim,
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
        self.to_out_prope = ReplicatedLinear(dim, dim, bias=True)
        nn.init.zeros_(self.to_out_prope.weight)
        if self.to_out_prope.bias is not None:
            nn.init.zeros_(self.to_out_prope.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None = None,
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

        # Self-attention with camera PRoPE
        attn_output_rope, attn_output_prope = self.attn1(
            query, key, value, freqs_cis,
            block_mask, kv_cache, current_start, cache_start,
            viewmats, Ks, is_cache=is_cache
        )
        # Combine rope and prope outputs
        attn_output_rope = attn_output_rope.flatten(2)
        attn_output_rope, _ = self.to_out(attn_output_rope)
        attn_output_prope = attn_output_prope.flatten(2)
        attn_output_prope, _ = self.to_out_prope(attn_output_prope)
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
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        # Cross-attention residual in bfloat16
        hidden_states = hidden_states + attn_output

        # norm3 for FFN input in float32
        norm_hidden_states = self.norm3(
            hidden_states.float(), c_shift_msa, c_scale_msa
        ).type_as(hidden_states)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states.to(orig_dtype))
        hidden_states = self.mlp_residual(hidden_states.float(), ff_output.float(), c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class CausalWanGameActionTransformer3DModel(BaseDiT):

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
        self.local_attn_size = config.local_attn_size
        self.inner_dim = inner_dim

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings
        self.condition_embedder = WanGameActionTimeImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            CausalWanGameActionTransformerBlock(
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
        self.block_mask = None
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3

        self.__post_init__()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
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
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)
        """
        orig_dtype = hidden_states.dtype
        if isinstance(encoder_hidden_states_image, list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]

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

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, action, encoder_hidden_states, encoder_hidden_states_image=encoder_hidden_states_image)

        # condition_embedder returns:
        # - temb: [B*T, dim] where T = post_patch_num_frames
        # - timestep_proj: [B*T, 6*dim]
        # Reshape to [B, T, 6, dim] for transformer blocks
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size))  # [B*T, 6, dim]
        timestep_proj = timestep_proj.view(batch_size, post_patch_num_frames, 6, self.hidden_size)  # [B, T, 6, dim]

        encoder_hidden_states = encoder_hidden_states_image

        # Transformer blocks
        for block_idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, freqs_cis,
                    self.block_mask,
                    kv_cache[block_idx] if kv_cache else None,
                    crossattn_cache[block_idx] if crossattn_cache else None,
                    current_start, cache_start,
                    viewmats, Ks, is_cache)
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, freqs_cis,
                    block_mask=self.block_mask,
                    kv_cache=kv_cache[block_idx] if kv_cache else None,
                    crossattn_cache=crossattn_cache[block_idx] if crossattn_cache else None,
                    current_start=current_start, cache_start=cache_start,
                    viewmats=viewmats, Ks=Ks, is_cache=is_cache)

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

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        action: torch.Tensor | None = None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        start_frame: int = 0,
        **kwargs
    ) -> torch.Tensor:

        orig_dtype = hidden_states.dtype
        if isinstance(encoder_hidden_states_image, list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]

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

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.dim() == 2:
            timestep = timestep.flatten()

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, action, encoder_hidden_states, encoder_hidden_states_image=encoder_hidden_states_image)

        # condition_embedder returns:
        # - temb: [B*T, dim] where T = post_patch_num_frames
        # - timestep_proj: [B*T, 6*dim]
        # Reshape to [B, T, 6, dim] for transformer blocks
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size))  # [B*T, 6, dim]
        timestep_proj = timestep_proj.view(batch_size, post_patch_num_frames, 6, self.hidden_size)  # [B, T, 6, dim]

        encoder_hidden_states = encoder_hidden_states_image

        # Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis,
                    self.block_mask,
                    None, None,  # kv_cache, crossattn_cache
                    0, None,     # current_start, cache_start
                    viewmats, Ks, False)  # viewmats, Ks, is_cache
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                      timestep_proj, freqs_cis,
                                      block_mask=self.block_mask,
                                      viewmats=viewmats, Ks=Ks)

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

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)
