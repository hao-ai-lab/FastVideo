# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, Optional, List
import math

import torch
# import torch._dynamo
# torch._dynamo.config.cache_size_limit = 128
# try:
#     torch._dynamo.config.recompile_limit = 128
# except AttributeError:
#     pass
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="default")
import torch.distributed as dist

from fastvideo.attention import LocalAttention
from fastvideo.forward_context import set_forward_context
from fastvideo.configs.models.dits import HunyuanVideo15Config
from fastvideo.layers.layernorm import (LayerNormScaleShift, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
# TODO(will-PY-refactor): RMSNorm ....
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed, _apply_rotary_emb
from fastvideo.layers.visual_embedding import (ModulateProjection, PatchEmbed,
                                               unpatchify)
from fastvideo.models.dits.base import CachableDiT
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.logger import init_logger

from fastvideo.models.dits.hunyuanvideo15 import (
    HunyuanRMSNorm, 
    HunyuanVideo15TimeEmbedding, 
    HunyuanVideo15ByT5TextProjection,
    HunyuanVideo15ImageProjection,
    SingleTokenRefiner,
    FinalLayer)

logger = init_logger(__name__)

class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal DiT block with separate modulation for text and image/video,
    using distributed attention and linear layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        local_attn_size: int = -1,
        sink_size: int = 0,
        dtype: torch.dtype | None = None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...]
        | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.deterministic = False
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Image modulation components
        self.img_mod = ModulateProjection(
            hidden_size,
            factor=6,
            act_layer="silu",
            dtype=dtype,
            prefix=f"{prefix}.img_mod",
        )

        # Fused operations for image stream
        self.img_attn_norm = LayerNormScaleShift(hidden_size,
                                                 norm_type="layer",
                                                 elementwise_affine=False,
                                                 dtype=dtype)
        self.img_attn_residual_mlp_norm = ScaleResidualLayerNormScaleShift(
            hidden_size,
            norm_type="layer",
            elementwise_affine=False,
            dtype=dtype)
        self.img_mlp_residual = ScaleResidual()

        # Image attention components
        self.img_attn_qkv = ReplicatedLinear(hidden_size,
                                             hidden_size * 3,
                                             bias=True,
                                             params_dtype=dtype,
                                             prefix=f"{prefix}.img_attn_qkv")

        self.img_attn_q_norm = HunyuanRMSNorm(self.head_dim, eps=1e-6, dtype=dtype)
        self.img_attn_k_norm = HunyuanRMSNorm(self.head_dim, eps=1e-6, dtype=dtype)

        self.img_attn_proj = ReplicatedLinear(hidden_size,
                                              hidden_size,
                                              bias=True,
                                              params_dtype=dtype,
                                              prefix=f"{prefix}.img_attn_proj")

        self.img_mlp = MLP(hidden_size,
                           mlp_hidden_dim,
                           bias=True,
                           dtype=dtype,
                           prefix=f"{prefix}.img_mlp")

        # Text modulation components
        self.txt_mod = ModulateProjection(
            hidden_size,
            factor=6,
            act_layer="silu",
            dtype=dtype,
            prefix=f"{prefix}.txt_mod",
        )

        # Fused operations for text stream
        self.txt_attn_norm = LayerNormScaleShift(hidden_size,
                                                 norm_type="layer",
                                                 elementwise_affine=False,
                                                 dtype=dtype)
        self.txt_attn_residual_mlp_norm = ScaleResidualLayerNormScaleShift(
            hidden_size,
            norm_type="layer",
            elementwise_affine=False,
            dtype=dtype)
        self.txt_mlp_residual = ScaleResidual()

        # Text attention components
        self.txt_attn_qkv = ReplicatedLinear(hidden_size,
                                             hidden_size * 3,
                                             bias=True,
                                             params_dtype=dtype)

        # QK norm layers for text
        self.txt_attn_q_norm = HunyuanRMSNorm(self.head_dim, eps=1e-6, dtype=dtype)
        self.txt_attn_k_norm = HunyuanRMSNorm(self.head_dim, eps=1e-6, dtype=dtype)

        self.txt_attn_proj = ReplicatedLinear(hidden_size,
                                              hidden_size,
                                              bias=True,
                                              params_dtype=dtype)

        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype)

        self.max_attention_size = 21 * 1590 if local_attn_size == -1 else local_attn_size * 1590

        self.attn = LocalAttention(
            num_heads=self.num_attention_heads,
            head_size=self.head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn"
        )

    def forward_txt(
        self,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cache_txt: bool = False,
    ):
        txt_mod_outputs = self.txt_mod(vec)
        (
            txt_attn_shift,
            txt_attn_scale,
            txt_attn_gate,
            txt_mlp_shift,
            txt_mlp_scale,
            txt_mlp_gate,
        ) = torch.chunk(txt_mod_outputs, 6, dim=-1)

        # Prepare text for attention using fused operation
        txt_attn_input = self.txt_attn_norm(txt, txt_attn_shift, txt_attn_scale)

        # Get QKV for text
        txt_qkv, _ = self.txt_attn_qkv(txt_attn_input)
        batch_size, text_seq_len = txt_qkv.shape[0], txt_qkv.shape[1]

        # Split QKV
        txt_qkv = txt_qkv.view(batch_size, text_seq_len, 3,
                               self.num_attention_heads, -1)
        txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :,
                                                                          2]
        # Apply QK-Norm if needed
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_q.dtype)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_k.dtype)

        t_kv = {}
        if cache_txt:
            t_kv["k_txt"] = txt_k
            t_kv["v_txt"] = txt_v

        txt_attn = self.attn(txt_q, txt_k, txt_v)
        # Process text attention output
        txt_attn_out, _ = self.txt_attn_proj(
            txt_attn.reshape(batch_size, text_seq_len, -1))

        # Use fused operation for residual connection, normalization, and modulation
        txt_mlp_input, txt_residual = self.txt_attn_residual_mlp_norm(
            txt, txt_attn_out, txt_attn_gate, txt_mlp_shift, txt_mlp_scale)

        # Process text MLP
        txt_mlp_out = self.txt_mlp(txt_mlp_input)
        txt = self.txt_mlp_residual(txt_residual, txt_mlp_out, txt_mlp_gate)

        return txt, t_kv

    def forward_vision(
        self,
        img: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple,
        block_mask: BlockMask,
        kv_cache: dict | None = None,
        txt_kv_cache: list | None = None,
        current_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process modulation vectors
        if vec.dim() == 3:
            img_mod_outputs = self.img_mod(vec).unflatten(dim=-1, sizes=(6, -1))
            (
                img_attn_shift,
                img_attn_scale,
                img_attn_gate,
                img_mlp_shift,
                img_mlp_scale,
                img_mlp_gate,
            ) = torch.chunk(img_mod_outputs, 6, dim=2)
        else:
            img_mod_outputs = self.img_mod(vec)
            (
                img_attn_shift,
                img_attn_scale,
                img_attn_gate,
                img_mlp_shift,
                img_mlp_scale,
                img_mlp_gate,
            ) = torch.chunk(img_mod_outputs, 6, dim=-1)

        # Prepare image for attention using fused operation
        img_attn_input = self.img_attn_norm(img, img_attn_shift, img_attn_scale)
        # Get QKV for image
        img_qkv, _ = self.img_attn_qkv(img_attn_input)
        batch_size, image_seq_len = img_qkv.shape[0], img_qkv.shape[1]

        # Split QKV
        img_qkv = img_qkv.view(batch_size, image_seq_len, 3,
                               self.num_attention_heads, -1)
        img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :,
                                                                          2]

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply rotary embeddings
        cos, sin = freqs_cis
        img_q = _apply_rotary_emb(img_q, cos, sin, is_neox_style=False)
        img_k = _apply_rotary_emb(img_k, cos, sin, is_neox_style=False)

        # Apply flex_attention
        # Does not support SP padding for now
        if kv_cache is None:
            q = img_q
            k = torch.cat([img_k, txt_kv_cache["k_txt"]], dim=1)
            v = torch.cat([img_v, txt_kv_cache["v_txt"]], dim=1)
            # Padding for flex attention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_kv_length = math.ceil(k.shape[1] / 128) * 128 - k.shape[1]
            padded_roped_query = torch.cat(
                [q,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [k, torch.zeros([k.shape[0], padded_kv_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_kv_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            img_attn = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)

            assert img_attn.shape[1] == image_seq_len
            updated_kv_cache = None
        else:
            current_end = current_start + img_q.shape[1]
            num_new_tokens = img_q.shape[1]
            sink_tokens = self.sink_size * 1590
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = self.max_attention_size
            
            # Clone cache to avoid in-place modification during gradient checkpointing
            k_cache = kv_cache["k"].clone()
            v_cache = kv_cache["v"].clone()
            
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache_size - num_new_tokens - sink_tokens
                k_cache[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    k_cache[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                v_cache[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    v_cache[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                assert local_end_index == self.max_attention_size
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

            assert local_start_index >= 0
            q = img_q
            k = torch.cat([k_cache[:, :local_start_index], img_k, txt_kv_cache["k_txt"]], dim=1)
            v = torch.cat([v_cache[:, :local_start_index], img_v, txt_kv_cache["v_txt"]], dim=1)
            img_attn = self.attn(q, k, v)

            k_cache[:, local_start_index:local_end_index] = img_k
            v_cache[:, local_start_index:local_end_index] = img_v
            
            updated_kv_cache = {
                "k": k_cache,
                "v": v_cache,
                "global_end_index": torch.tensor([current_end], dtype=torch.long, device=k_cache.device),
                "local_end_index": torch.tensor([local_end_index], dtype=torch.long, device=k_cache.device)
            }
        
        img_attn_out, _ = self.img_attn_proj(
            img_attn.view(batch_size, image_seq_len, -1))
        # Use fused operation for residual connection, normalization, and modulation
        img_mlp_input, img_residual = self.img_attn_residual_mlp_norm(
            img, img_attn_out, img_attn_gate, img_mlp_shift, img_mlp_scale)

        # Process image MLP
        img_mlp_out = self.img_mlp(img_mlp_input)
        img = self.img_mlp_residual(img_residual, img_mlp_out, img_mlp_gate)

        return img, updated_kv_cache

    def forward(
        self,
        txt_inference=False,
        vision_inference=False,
        **kwargs
    ):
        if txt_inference:
            return self.forward_txt(**kwargs)
        elif vision_inference:
            return self.forward_vision(**kwargs)
        else:
            raise ValueError("txt_inference and vision_inference cannot be both False")


class CausalHunyuanVideo15Transformer3DModel(CachableDiT):
    r"""
    A Transformer model for video-like data used in [HunyuanVideo1.5](https://huggingface.co/tencent/HunyuanVideo1.5).
    """

    # shard single stream, double stream blocks, and refiner_blocks
    _fsdp_shard_conditions = HunyuanVideo15Config()._fsdp_shard_conditions
    _compile_conditions = HunyuanVideo15Config()._compile_conditions
    _supported_attention_backends = HunyuanVideo15Config(
    )._supported_attention_backends
    param_names_mapping = HunyuanVideo15Config().param_names_mapping
    reverse_param_names_mapping = HunyuanVideo15Config(
    ).reverse_param_names_mapping
    lora_param_names_mapping = HunyuanVideo15Config().lora_param_names_mapping

    def __init__(
        self,
        config: HunyuanVideo15Config,
        hf_config: dict[str, Any],
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.out_channels = config.out_channels or config.in_channels
        self.patch_size = (config.patch_size_t, config.patch_size, config.patch_size)

        # 1. Latent and condition embedders
        self.img_in = PatchEmbed(self.patch_size, 
                                    config.in_channels, 
                                    self.hidden_size,
                                    prefix=f"{config.prefix}.img_in")
        self.image_embedder = HunyuanVideo15ImageProjection(config.image_embed_dim, self.hidden_size)

        self.txt_in = SingleTokenRefiner(config.text_embed_dim,
                                                   self.hidden_size,
                                                   config.num_attention_heads,
                                                   depth=config.num_refiner_layers,
                                                   dtype=None,
                                                   prefix=f"{config.prefix}.txt_in")

        self.txt_in_2 = HunyuanVideo15ByT5TextProjection(config.text_embed_2_dim, 2048, self.hidden_size)

        self.time_in = HunyuanVideo15TimeEmbedding(self.hidden_size, use_meanflow=config.use_meanflow)

        self.cond_type_embed = nn.Embedding(3, self.hidden_size)

        # 3. Dual stream transformer blocks

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    local_attn_size=config.local_attn_size,
                    sink_size=config.sink_size,
                    dtype=None,
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=f"{config.prefix}.double_blocks.{i}"
                )
                for i in range(config.num_layers)
            ]
        )

        # 5. Output projection
        self.final_layer = FinalLayer(self.hidden_size,
                                self.patch_size,
                                self.out_channels,
                                prefix=f"{config.prefix}.final_layer")

        self.gradient_checkpointing = False

        self.num_frame_per_block = config.num_frames_per_block
        self.local_attn_size = config.local_attn_size
        self.block_mask = None

        self.__post_init__()

    def get_text_and_mask(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_attention_mask_2: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor,
        timestep: torch.Tensor,
    ):
        batch_size, txt_seq_len = encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]
        # qwen text embedding
        encoder_hidden_states = self.txt_in(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        # byt5 text embedding
        encoder_hidden_states_2 = self.txt_in_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        # image embed
        encoder_hidden_states_3 = self.image_embedder(encoder_hidden_states_image)
        is_t2v = torch.all(encoder_hidden_states_image == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2
            * torch.ones_like(
                encoder_hidden_states_3[:, :, 0],
                dtype=torch.long,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        # reorder and combine text tokens: combine valid tokens first, then padding
        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            # Concatenate: [valid_image, valid_byt5, valid_mllm, invalid_image, invalid_byt5, invalid_mllm]
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],  # valid image
                        text_2[text_mask_2],  # valid byt5
                        text[text_mask],  # valid mllm
                        image[~image_mask],  # invalid image (zeroed)
                        torch.zeros_like(text_2[~text_mask_2]),  # invalid byt5 (zeroed)
                        torch.zeros_like(text[~text_mask]),  # invalid mllm (zeroed)
                    ],
                    dim=0,
                )
            )
            # Apply same reordering to attention masks
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)
        assert encoder_hidden_states.shape[0] == 1
        return encoder_hidden_states, encoder_attention_mask

    
    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1,
        text_seq_len: int = 0
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen
        total_kv_length = total_length + text_seq_len

        total_length_tensor = torch.tensor(total_length, device=device)
        total_kv_length_tensor = torch.tensor(total_kv_length, device=device)

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        kv_padded_length = math.ceil(total_kv_length / 128) * 128 - total_kv_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            # start=frame_seqlen,
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        # frame_indices = torch.cat([torch.tensor([0], device=device), frame_indices])

        for i, tmp in enumerate(frame_indices):
            # if i == 0:
            #     ends[tmp:tmp + frame_seqlen] = tmp + frame_seqlen
            # else:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx) | ((kv_idx >= total_length_tensor) & (kv_idx < total_kv_length_tensor))
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx) | ((kv_idx >= total_length_tensor) & (kv_idx < total_kv_length_tensor))
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_kv_length + kv_padded_length, _compile=False, device=device)

        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     print(
        #         f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
        #     print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def forward_txt(
        self,
        encoder_hidden_states: List[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: Optional[List[torch.Tensor]] = None,
        encoder_attention_mask: Optional[List[torch.Tensor]] = None,
        guidance: Optional[torch.Tensor] = None,
        timestep_r: Optional[torch.LongTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        cache_txt: bool = False,
        **kwargs
    ):
        # Check that the timestep is only consisted of 0s
        assert torch.all(timestep == 0), "Timestep for txt must be only consisted of 0s"

        if cache_txt:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for _ in range(transformer_num_layers):
                _kv_cache_new.append(
                    {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
                )

        encoder_hidden_states_image = encoder_hidden_states_image[0]
        encoder_hidden_states, encoder_hidden_states_2 = encoder_hidden_states
        encoder_attention_mask, encoder_attention_mask_2 = encoder_attention_mask

        # 2. Conditional embeddings
        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            temb = self.time_in(timestep.flatten(), timestep_r=timestep_r, timestep_seq_len=ts_seq_len)
        else:
            temb = self.time_in(timestep, timestep_r=timestep_r)
        # temb is [bs, seq_len, inner_dim] if ts_seq_len is not None, otherwise [bs, inner_dim]

        encoder_hidden_states, encoder_attention_mask = self.get_text_and_mask(
            encoder_hidden_states,
            encoder_hidden_states_2,
            encoder_attention_mask,
            encoder_attention_mask_2,
            encoder_hidden_states_image,
            timestep
        )
        
        encoder_hidden_states = encoder_hidden_states[encoder_attention_mask.bool().to(encoder_hidden_states.device)].unsqueeze(0)

        # 4. Transformer blocks
        for index, block in enumerate(self.double_blocks):
            encoder_hidden_states, t_kv = block(
                txt_inference=True,
                vision_inference=False,
                txt=encoder_hidden_states,
                vec=temb,
                cache_txt=cache_txt,
            )

            if cache_txt:
                _kv_cache_new[index]["k_txt"] = t_kv["k_txt"]
                _kv_cache_new[index]["v_txt"] = t_kv["v_txt"]

        if cache_txt:
            return _kv_cache_new

    def forward_vision(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        guidance: Optional[torch.Tensor] = None,
        timestep_r: Optional[torch.LongTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        kv_cache: dict | None = None,
        txt_kv_cache: list | None = None,
        current_start: int = 0,
        rope_start_idx: int = 0,
    ):
        assert txt_kv_cache is not None, "txt_kv_cache must be provided"

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        # Get rotary embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height, post_patch_width), self.hidden_size,
            self.num_attention_heads, self.config.rope_axes_dim, self.config.rope_theta, start_frame=rope_start_idx)
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # 2. Conditional embeddings
        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            temb = self.time_in(timestep.flatten(), timestep_r=timestep_r, timestep_seq_len=ts_seq_len)
        else:
            temb = self.time_in(timestep, timestep_r=timestep_r)
        # temb is [bs, seq_len, inner_dim] if ts_seq_len is not None, otherwise [bs, inner_dim]

        hidden_states = self.img_in(hidden_states)

        # Prepare block-wise causal attention mask
        if kv_cache is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                text_seq_len=txt_kv_cache[0]["k_txt"].shape[1]
            )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block_index, block in enumerate(self.double_blocks):
                hidden_states, new_cache = self._gradient_checkpointing_func(
                    block,
                    txt_inference=False,
                    vision_inference=True,
                    img=hidden_states,
                    vec=temb,
                    freqs_cis=freqs_cis,
                    block_mask=self.block_mask,
                    kv_cache=kv_cache[block_index] if kv_cache is not None else None,
                    txt_kv_cache=txt_kv_cache[block_index],
                    current_start=current_start
                )
                if new_cache is not None and kv_cache is not None:
                    for k in new_cache.keys():
                        kv_cache[block_index][k] = new_cache[k].clone()

        else:
            for block_index, block in enumerate(self.double_blocks):
                hidden_states, new_cache = block(
                    txt_inference=False,
                    vision_inference=True,
                    img=hidden_states,
                    vec=temb,
                    freqs_cis=freqs_cis,
                    block_mask=self.block_mask,
                    kv_cache=kv_cache[block_index] if kv_cache is not None else None,
                    txt_kv_cache=txt_kv_cache[block_index],
                    current_start=current_start
                )
                if new_cache is not None and kv_cache is not None:
                    for k in new_cache.keys():
                        kv_cache[block_index][k] = new_cache[k].clone()

        # Final layer processing
        hidden_states = self.final_layer(hidden_states, temb)
        # Unpatchify to get original shape
        hidden_states = unpatchify(hidden_states, post_patch_num_frames, post_patch_height, post_patch_width, self.patch_size, self.out_channels)

        return hidden_states, kv_cache

    def forward(
        self,
        txt_inference=False,
        vision_inference=False,
        **kwargs,
    ):
        if txt_inference:
            return self.forward_txt(**kwargs)
        elif vision_inference:
            return self.forward_vision(**kwargs)
        else:
            raise ValueError("txt_inference and vision_inference cannot be both False")