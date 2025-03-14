# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward as HF_FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import MochiCombinedTimestepCaptionEmbedding, PatchEmbed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.models.mochi_hf.norm import (MochiLayerNormContinuous, MochiModulatedRMSNorm, MochiRMSNorm,
                                            MochiRMSNormZero)
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FeedForward(HF_FeedForward):

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__(dim, dim_out, mult, dropout, activation_fn, final_dropout, inner_dim, bias)
        assert activation_fn == "swiglu"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0].proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)

        return self.net[2](LigerSiLUMulFunction.apply(gate, hidden_states))


class MochiAttention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        processor: "MochiAttnProcessor2_0",
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_dim: int = None,
        out_context_dim: int = None,
        out_bias: bool = True,
        context_pre_only: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim else query_dim
        self.context_pre_only = context_pre_only

        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.norm_q = MochiRMSNorm(dim_head, eps)
        self.norm_k = MochiRMSNorm(dim_head, eps)
        self.norm_added_q = MochiRMSNorm(dim_head, eps)
        self.norm_added_k = MochiRMSNorm(dim_head, eps)

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        if self.context_pre_only is not None:
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)

        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


class MochiAttnProcessor2_0:
    """Attention processor used in Mochi."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MochiAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [b, s, h * d]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # [b, s, h=24, d=128]
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # [b, 256, h * d]
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        # [b, 256, h=24, d=128]
        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb[0], image_rotary_emb[1]
        # shard the head dimension
        if get_sequence_parallel_state():
            # B, S, H, D to (S, B,) H, D
            # batch_size, seq_len, attn_heads, head_dim
            query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
            key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
            value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)

            def shrink_head(encoder_state, dim):
                local_heads = encoder_state.shape[dim] // nccl_info.sp_size
                return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads, local_heads)

            encoder_query = shrink_head(encoder_query, dim=2)
            encoder_key = shrink_head(encoder_key, dim=2)
            encoder_value = shrink_head(encoder_value, dim=2)
            if image_rotary_emb is not None:
                freqs_cos = shrink_head(freqs_cos, dim=1)
                freqs_sin = shrink_head(freqs_sin, dim=1)

        if image_rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()
                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)

            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        # query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        # encoder_query, encoder_key, encoder_value = (
        #     encoder_query.transpose(1, 2),
        #     encoder_key.transpose(1, 2),
        #     encoder_value.transpose(1, 2),
        # )
        # [b, s, h, d]
        sequence_length = query.size(1)
        encoder_sequence_length = encoder_query.size(1)

        # H
        query = torch.cat([query, encoder_query], dim=1).unsqueeze(2)
        key = torch.cat([key, encoder_key], dim=1).unsqueeze(2)
        value = torch.cat([value, encoder_value], dim=1).unsqueeze(2)
        # B, S, 3, H, D
        qkv = torch.cat([query, key, value], dim=2)

        attn_mask = encoder_attention_mask[:, :].bool()
        attn_mask = F.pad(attn_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask = None, dropout_p=0.0, is_causal=False)

        # valid_lengths = encoder_attention_mask.sum(dim=1) + sequence_length
        # def no_padding_mask(score, b, h, q_idx, kv_idx):
        #     return torch.where(kv_idx < valid_lengths[b],score,  -float("inf"))

        # hidden_states = flex_attention(query, key, value, score_mod=no_padding_mask)
        if get_sequence_parallel_state():
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                (sequence_length, encoder_sequence_length), dim=1)
            # B, S, H, D
            hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
            encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)
            encoder_hidden_states = encoder_hidden_states.flatten(2, 3)
            encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        else:
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                (sequence_length, encoder_sequence_length), dim=1)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class MochiTransformerBlock(nn.Module):
    r"""
    Transformer block used in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"swiglu"`):
            Activation function to use in feed-forward.
        context_pre_only (`bool`, defaults to `False`):
            Whether or not to process context-related conditions with additional layers.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        pooled_projection_dim: int,
        qk_norm: str = "rms_norm",
        activation_fn: str = "swiglu",
        context_pre_only: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.context_pre_only = context_pre_only
        self.ff_inner_dim = (4 * dim * 2) // 3
        self.ff_context_inner_dim = (4 * pooled_projection_dim * 2) // 3

        self.norm1 = MochiRMSNormZero(dim, 4 * dim, eps=eps, elementwise_affine=False)

        if not context_pre_only:
            self.norm1_context = MochiRMSNormZero(dim, 4 * pooled_projection_dim, eps=eps, elementwise_affine=False)
        else:
            self.norm1_context = MochiLayerNormContinuous(
                embedding_dim=pooled_projection_dim,
                conditioning_embedding_dim=dim,
                eps=eps,
            )

        self.attn1 = MochiAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=False,
            added_kv_proj_dim=pooled_projection_dim,
            added_proj_bias=False,
            out_dim=dim,
            out_context_dim=pooled_projection_dim,
            context_pre_only=context_pre_only,
            processor=MochiAttnProcessor2_0(),
            eps=1e-5,
        )

        # TODO(aryan): norm_context layers are not needed when `context_pre_only` is True
        self.norm2 = MochiModulatedRMSNorm(eps=eps)
        self.norm2_context = (MochiModulatedRMSNorm(eps=eps) if not self.context_pre_only else None)

        self.norm3 = MochiModulatedRMSNorm(eps)
        self.norm3_context = (MochiModulatedRMSNorm(eps=eps) if not self.context_pre_only else None)

        self.ff = FeedForward(dim, inner_dim=self.ff_inner_dim, activation_fn=activation_fn, bias=False)
        self.ff_context = None
        if not context_pre_only:
            self.ff_context = FeedForward(
                pooled_projection_dim,
                inner_dim=self.ff_context_inner_dim,
                activation_fn=activation_fn,
                bias=False,
            )

        self.norm4 = MochiModulatedRMSNorm(eps=eps)
        self.norm4_context = MochiModulatedRMSNorm(eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
        output_attn=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

        if not self.context_pre_only:
            (
                norm_encoder_hidden_states,
                enc_gate_msa,
                enc_scale_mlp,
                enc_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)

        attn_hidden_states, context_attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = hidden_states + self.norm2(attn_hidden_states, torch.tanh(gate_msa).unsqueeze(1))
        norm_hidden_states = self.norm3(hidden_states, (1 + scale_mlp.unsqueeze(1).to(torch.float32)))
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + self.norm4(ff_output, torch.tanh(gate_mlp).unsqueeze(1))

        if not self.context_pre_only:
            encoder_hidden_states = encoder_hidden_states + self.norm2_context(context_attn_hidden_states,
                                                                               torch.tanh(enc_gate_msa).unsqueeze(1))
            norm_encoder_hidden_states = self.norm3_context(
                encoder_hidden_states,
                (1 + enc_scale_mlp.unsqueeze(1).to(torch.float32)),
            )
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + self.norm4_context(context_ff_output,
                                                                               torch.tanh(enc_gate_mlp).unsqueeze(1))

        if not output_attn:
            attn_hidden_states = None
        return hidden_states, encoder_hidden_states, attn_hidden_states


class MochiRoPE(nn.Module):
    r"""
    RoPE implementation used in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        base_height (`int`, defaults to `192`):
            Base height used to compute interpolation scale for rotary positional embeddings.
        base_width (`int`, defaults to `192`):
            Base width used to compute interpolation scale for rotary positional embeddings.
    """

    def __init__(self, base_height: int = 192, base_width: int = 192) -> None:
        super().__init__()

        self.target_area = base_height * base_width

    def _centers(self, start, stop, num, device, dtype) -> torch.Tensor:
        edges = torch.linspace(start, stop, num + 1, device=device, dtype=dtype)
        return (edges[:-1] + edges[1:]) / 2

    def _get_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        scale = (self.target_area / (height * width))**0.5
        t = torch.arange(num_frames * nccl_info.sp_size, device=device, dtype=dtype)
        h = self._centers(-height * scale / 2, height * scale / 2, height, device, dtype)
        w = self._centers(-width * scale / 2, width * scale / 2, width, device, dtype)

        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")

        positions = torch.stack([grid_t, grid_h, grid_w], dim=-1).view(-1, 3)
        return positions

    def _create_rope(self, freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        with torch.autocast(freqs.device.type, enabled=False):
            # Always run ROPE freqs computation in FP32
            freqs = torch.einsum(
                "nd,dhf->nhf",  # codespell:ignore
                pos.to(torch.float32),  # codespell:ignore
                freqs.to(torch.float32))
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    def forward(
        self,
        pos_frequencies: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self._get_positions(num_frames, height, width, device, dtype)
        rope_cos, rope_sin = self._create_rope(pos_frequencies, pos)
        return rope_cos, rope_sin


@maybe_allow_in_graph
class MochiTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data introduced in [Mochi](https://huggingface.co/genmo/mochi-1-preview).

    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `48`):
            The number of layers of Transformer blocks to use.
        in_channels (`int`, defaults to `12`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `256`):
            Output dimension of timestep embeddings.
        activation_fn (`str`, defaults to `"swiglu"`):
            Activation function to use in feed-forward.
        max_sequence_length (`int`, defaults to `256`):
            The maximum sequence length of text embeddings supported.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 48,
        pooled_projection_dim: int = 1536,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 4096,
        time_embed_dim: int = 256,
        activation_fn: str = "swiglu",
        max_sequence_length: int = 256,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            pos_embed_type=None,
        )

        self.time_embed = MochiCombinedTimestepCaptionEmbedding(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            text_embed_dim=text_embed_dim,
            time_embed_dim=time_embed_dim,
            num_attention_heads=8,
        )

        self.pos_frequencies = nn.Parameter(torch.full((3, num_attention_heads, attention_head_dim // 2), 0.0))
        self.rope = MochiRoPE()

        self.transformer_blocks = nn.ModuleList([
            MochiTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                pooled_projection_dim=pooled_projection_dim,
                qk_norm=qk_norm,
                activation_fn=activation_fn,
                context_pre_only=i == num_layers - 1,
            ) for i in range(num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(
            inner_dim,
            inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            norm_type="layer_norm",
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        assert (return_dict is False), "return_dict is not supported in MochiTransformer3DModel"

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (attention_kwargs is not None and attention_kwargs.get("scale", None) is not None):
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size

        post_patch_height = height // p
        post_patch_width = width // p
        # Peiyuan: This is hacked to force mochi to follow the behaviour of SD3 and Flux
        timestep = 1000 - timestep
        temb, encoder_hidden_states = self.time_embed(
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_dtype=hidden_states.dtype,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        image_rotary_emb = self.rope(
            self.pos_frequencies,
            num_frames,
            post_patch_height,
            post_patch_width,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        attn_outputs_list = []
        for i, block in enumerate(self.transformer_blocks):
            if self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = ({"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {})
                (
                    hidden_states,
                    encoder_hidden_states,
                    attn_outputs,
                ) = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    temb,
                    image_rotary_emb,
                    output_features,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states, attn_outputs = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    output_attn=output_features,
                )
            if i % output_features_stride == 0:
                attn_outputs_list.append(attn_outputs)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not output_features:
            attn_outputs_list = None
        else:
            attn_outputs_list = torch.stack(attn_outputs_list, dim=0)
        # Peiyuan: This is hacked to force mochi to follow the behaviour of SD3 and Flux
        return (-output, attn_outputs_list)
