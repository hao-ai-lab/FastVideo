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
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="default")
import torch.distributed as dist

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard)
from fastvideo.configs.models.dits import HunyuanVideo15Config
from fastvideo.layers.layernorm import (LayerNormScaleShift, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
# TODO(will-PY-refactor): RMSNorm ....
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed, _apply_rotary_emb
from fastvideo.layers.visual_embedding import (ModulateProjection, PatchEmbed,
                                               TimestepEmbedder, unpatchify)
from fastvideo.models.dits.base import CachableDiT
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.logger import init_logger
from fastvideo.forward_context import set_forward_context

from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.distributed.utils import create_attention_mask_for_padding

from hunyuanvideo15 import (
    HunyuanRMSNorm, 
    HunyuanVideo15TimeEmbedding, 
    HunyuanVideo15ByT5TextProjection,
    HunyuanVideo15ImageProjection)

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
        dtype: torch.dtype | None = None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...]
        | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.deterministic = False
        self.num_attention_heads = num_attention_heads
        head_dim = hidden_size // num_attention_heads
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

        self.img_attn_q_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.img_attn_k_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)

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
        self.txt_attn_q_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.txt_attn_k_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)

        self.txt_attn_proj = ReplicatedLinear(hidden_size,
                                              hidden_size,
                                              bias=True,
                                              params_dtype=dtype)

        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype)

        # Distributed attention
        self.attn = DistributedAttention(
            num_heads=num_attention_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn")

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple,
        block_mask: BlockMask,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process modulation vectors
        img_mod_outputs = self.img_mod(vec)
        (
            img_attn_shift,
            img_attn_scale,
            img_attn_gate,
            img_mlp_shift,
            img_mlp_scale,
            img_mlp_gate,
        ) = torch.chunk(img_mod_outputs, 6, dim=-1)

        txt_mod_outputs = self.txt_mod(vec)
        (
            txt_attn_shift,
            txt_attn_scale,
            txt_attn_gate,
            txt_mlp_shift,
            txt_mlp_scale,
            txt_mlp_gate,
        ) = torch.chunk(txt_mod_outputs, 6, dim=-1)

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

        # Apply flex_attention
        # Does not support SP padding for now
        if kv_cache is None:
            txt_seq_len = txt_q.shape[1]
            q = torch.cat([img_q, txt_q], dim=1)
            k = torch.cat([img_k, txt_k], dim=1)
            v = torch.cat([img_v, txt_v], dim=1)
            # Padding for flex attention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [q,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            attn_out = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)

            assert attn_out.shape[1] == txt_seq_len + image_seq_len
            img_attn = attn_out[:, :image_seq_len, :]
            txt_attn = attn_out[:, image_seq_len:, :]
        
        img_attn_out, _ = self.img_attn_proj(
            img_attn.view(batch_size, image_seq_len, -1))
        # Use fused operation for residual connection, normalization, and modulation
        img_mlp_input, img_residual = self.img_attn_residual_mlp_norm(
            img, img_attn_out, img_attn_gate, img_mlp_shift, img_mlp_scale)

        # Process image MLP
        img_mlp_out = self.img_mlp(img_mlp_input)
        img = self.img_mlp_residual(img_residual, img_mlp_out, img_mlp_gate)

        # Process text attention output
        txt_attn_out, _ = self.txt_attn_proj(
            txt_attn.reshape(batch_size, text_seq_len, -1))

        # Use fused operation for residual connection, normalization, and modulation
        txt_mlp_input, txt_residual = self.txt_attn_residual_mlp_norm(
            txt, txt_attn_out, txt_attn_gate, txt_mlp_shift, txt_mlp_scale)

        # Process text MLP
        txt_mlp_out = self.txt_mlp(txt_mlp_input)
        txt = self.txt_mlp_residual(txt_residual, txt_mlp_out, txt_mlp_gate)

        return img, txt


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

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1,
        txt_attn_mask: torch.Tensor = None
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        assert num_frames == 19
        img_seq_len = num_frames * frame_seqlen
        batch_size, txt_seq_len = txt_attn_mask.shape
        total_len = img_seq_len + txt_seq_len

        # we do right padding to get to a multiple of 128
        padded_total_len = math.ceil(total_len / 128) * 128

        ends = torch.zeros(img_seq_len,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=img_seq_len,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        frame_indices = torch.cat([torch.tensor([0], device=device), frame_indices])

        for i, tmp in enumerate(frame_indices):
            if i == 0:
                ends[tmp:tmp + frame_seqlen] = tmp + frame_seqlen
            else:
                ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                    frame_seqlen * num_frame_per_block

        # Generate the 2D text attention mask
        attention_mask_bool = txt_attn_mask.bool()
        self_attn_mask_1 = attention_mask_bool.view(batch_size, 1, 1, txt_seq_len)
        self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
        txt_attn_2d_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

        # Flatten tensors for safer indexing under vmap to avoid .item() calls
        txt_attn_mask_flat = attention_mask_bool.reshape(-1)
        txt_attn_2d_mask_flat = txt_attn_2d_mask.reshape(-1)
        ends_flat = ends.long()
        
        def attention_mask(b, h, q_idx, kv_idx):
            # Use flattened indexing to avoid vmap issues with multi-dimensional indexing
            q_idx_long = q_idx.long()
            kv_idx_long = kv_idx.long()
            b_long = b.long()

            q_is_img = q_idx_long < img_seq_len
            kv_is_img = kv_idx_long < img_seq_len
            
            q_idx_img_safe = q_idx_long.clamp(0, img_seq_len - 1)
            kv_idx_img_safe = kv_idx_long.clamp(0, img_seq_len - 1)
            
            q_idx_txt_safe = (q_idx_long - img_seq_len).clamp(0, txt_seq_len - 1)
            kv_idx_txt_safe = (kv_idx_long - img_seq_len).clamp(0, txt_seq_len - 1)

            # Case 1: Image attending to image
            q_end = ends_flat[q_idx_img_safe]
            if local_attn_size == -1:
                res_img_img = (kv_idx_img_safe < q_end) | (q_idx_img_safe == kv_idx_img_safe)
            else:
                res_img_img = ((kv_idx_img_safe < q_end) & (kv_idx_img_safe >= (q_end - local_attn_size * frame_seqlen))) | (q_idx_img_safe == kv_idx_img_safe)

            # Case 2: Text attending to text
            # txt_attn_2d_mask_flat index: b * 1 * txt_seq_len * txt_seq_len + 0 * txt_seq_len * txt_seq_len + q * txt_seq_len + kv
            idx_txt_txt = b_long * (txt_seq_len * txt_seq_len) + q_idx_txt_safe * txt_seq_len + kv_idx_txt_safe
            res_txt_txt = txt_attn_2d_mask_flat[idx_txt_txt]
            
            # Case 3: Image attending to text
            # txt_attn_mask_flat index: b * txt_seq_len + kv
            idx_img_txt = b_long * txt_seq_len + kv_idx_txt_safe
            res_img_txt = txt_attn_mask_flat[idx_img_txt]
            
            # Case 4: Text attending to image
            # index: b * txt_seq_len + q
            idx_txt_img = b_long * txt_seq_len + q_idx_txt_safe
            res_txt_img = txt_attn_mask_flat[idx_txt_img]

            mask = torch.where(
                q_is_img,
                torch.where(kv_is_img, res_img_img, res_img_txt),
                torch.where(kv_is_img, res_txt_img, res_txt_txt)
            )
            
            return mask & (q_idx_long < total_len) & (kv_idx_long < total_len)

        block_mask = create_block_mask(attention_mask, B=batch_size, H=None, Q_LEN=padded_total_len,
                                       KV_LEN=padded_total_len, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: List[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: List[torch.Tensor],
        encoder_attention_mask: List[torch.Tensor],
        guidance: Optional[torch.Tensor] = None,
        timestep_r: Optional[torch.LongTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        encoder_hidden_states_image = encoder_hidden_states_image[0]
        encoder_hidden_states, encoder_hidden_states_2 = encoder_hidden_states
        encoder_attention_mask, encoder_attention_mask_2 = encoder_attention_mask

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        # Get rotary embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height, post_patch_width), self.hidden_size,
            self.num_attention_heads, self.config.rope_axes_dim, self.config.rope_theta)
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # 2. Conditional embeddings
        temb = self.time_in(timestep, timestep_r=timestep_r)

        hidden_states = self.img_in(hidden_states)
        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)

        current_seq_len = hidden_states.shape[1]
        sp_world_size = get_sp_world_size()
        padded_seq_len = current_seq_len * sp_world_size
        
        if padded_seq_len > original_seq_len:
            seq_attention_mask = create_attention_mask_for_padding(
                seq_len=original_seq_len,
                padded_seq_len=padded_seq_len,
                batch_size=batch_size,
                device=hidden_states.device,
            )
        else:
            seq_attention_mask = None

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

        # Prepare block-wise causal attention mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                txt_attn_mask=encoder_attention_mask
            )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.double_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    freqs_cis,
                    self.block_mask
                )

        else:
            for block in self.double_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    freqs_cis,
                    self.block_mask
                )

        # Final layer processing
        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states = self.final_layer(hidden_states, temb)
        # Unpatchify to get original shape
        hidden_states = unpatchify(hidden_states, post_patch_num_frames, post_patch_height, post_patch_width, self.patch_size, self.out_channels)

        return hidden_states

class SingleTokenRefiner(nn.Module):
    """
    A token refiner that processes text embeddings with attention to improve
    their representation for cross-attention with image features.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        num_attention_heads,
        depth=2,
        qkv_bias=True,
        dtype=None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Input projection
        # self.input_embedder = ReplicatedLinear(
        #     in_channels,
        #     hidden_size,
        #     bias=True,
        #     params_dtype=dtype,
        #     prefix=f"{prefix}.input_embedder")
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size,
                                           act_layer="silu",
                                           dtype=dtype,
                                           prefix=f"{prefix}.t_embedder")

        # Context embedding
        self.c_embedder = MLP(in_channels,
                              hidden_size,
                              hidden_size,
                              act_type="silu",
                              dtype=dtype,
                              prefix=f"{prefix}.c_embedder")

        # Refiner blocks
        self.refiner_blocks = nn.ModuleList([
            IndividualTokenRefinerBlock(
                hidden_size,
                num_attention_heads,
                qkv_bias=qkv_bias,
                dtype=dtype,
                prefix=f"{prefix}.refiner_blocks.{i}",
            ) for i in range(depth)
        ])

    def forward(self, x, t, mask=None):
        # Get timestep embeddings
        timestep_aware_representations = self.t_embedder(t)

        # Get context-aware representations
        original_dtype = x.dtype
        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [B, L, 1]
            context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)

        context_aware_representations = self.c_embedder(
            context_aware_representations)
        c = timestep_aware_representations + context_aware_representations
        # Project input
        x = self.input_embedder(x)
        # Process through refiner blocks
        for block in self.refiner_blocks:
            x = block(x, c, mask)
        return x

class IndividualTokenRefinerBlock(nn.Module):
    """
    A transformer block for refining individual tokens with self-attention.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dtype=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Normalization and attention
        self.norm1 = nn.LayerNorm(hidden_size,
                                  eps=1e-6,
                                  elementwise_affine=True,
                                  dtype=dtype)

        self.self_attn_qkv = ReplicatedLinear(hidden_size,
                                              hidden_size * 3,
                                              bias=qkv_bias,
                                              params_dtype=dtype,
                                              prefix=f"{prefix}.self_attn_qkv")

        self.self_attn_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            params_dtype=dtype,
            prefix=f"{prefix}.self_attn_proj")

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size,
                                  eps=1e-6,
                                  elementwise_affine=True,
                                  dtype=dtype)
        self.mlp = MLP(hidden_size,
                       mlp_hidden_dim,
                       bias=True,
                       act_type="silu",
                       dtype=dtype,
                       prefix=f"{prefix}.mlp")

        # Modulation
        self.adaLN_modulation = ModulateProjection(
            hidden_size,
            factor=2,
            act_layer="silu",
            dtype=dtype,
            prefix=f"{prefix}.adaLN_modulation")

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_attention_heads,
            head_size=hidden_size // num_attention_heads,
            # TODO: remove hardcode; remove STA
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA),
        )

    def forward(self, x, c, mask=None):
        if mask is not None:
            mask = mask.clone().bool()
            mask[:, 0] = True  # Prevent attention weights from becoming NaN

        # Get modulation parameters
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=-1)
        # Self-attention
        norm_x = self.norm1(x)
        qkv, _ = self.self_attn_qkv(norm_x)

        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_attention_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Run scaled dot product attention
        from fastvideo.attention.backends.flash_attn import FlashAttnMetadataBuilder
        attn_metadata = FlashAttnMetadataBuilder().build(
            current_timestep=0,
            attn_mask=mask,
        )
        # Run distributed attention
        with set_forward_context(current_timestep=0, attn_metadata=attn_metadata):
            attn_output = self.attn(q, k, v)  # [B, L, H, D]
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          -1)  # [B, L, H*D]

        # Project and apply residual connection with gating
        attn_out, _ = self.self_attn_proj(attn_output)
        x = x + attn_out * gate_msa.unsqueeze(1)

        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out * gate_mlp.unsqueeze(1)

        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT that projects features to pixel space.
    """

    def __init__(self,
                 hidden_size,
                 patch_size,
                 out_channels,
                 dtype=None,
                 prefix: str = "") -> None:
        super().__init__()

        # Normalization
        self.norm_final = nn.LayerNorm(hidden_size,
                                       eps=1e-6,
                                       elementwise_affine=False,
                                       dtype=dtype)

        output_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_channels

        self.linear = ReplicatedLinear(hidden_size,
                                       output_dim,
                                       bias=True,
                                       params_dtype=dtype,
                                       prefix=f"{prefix}.linear")

        # Modulation
        self.adaLN_modulation = ModulateProjection(
            hidden_size,
            factor=2,
            act_layer="silu",
            dtype=dtype,
            prefix=f"{prefix}.adaLN_modulation")

    def forward(self, x, c):
        # What the heck HF? Why you change the scale and shift order here???
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x, _ = self.linear(x)
        return x