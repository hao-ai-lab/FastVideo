# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard)
from fastvideo.configs.models.dits import HyWorldConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import TimestepEmbedder, unpatchify
from fastvideo.models.dits.hunyuanvideo15 import (
    MMDoubleStreamBlock,
    HunyuanVideo15Transformer3DModel,
)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.logger import init_logger
from fastvideo.forward_context import set_forward_context
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.distributed.utils import create_attention_mask_for_padding

from .camera_rope import prope_qkv

logger = init_logger(__name__)


class HyWorldDoubleStreamBlock(MMDoubleStreamBlock):
    """
    Extended MMDoubleStreamBlock with ProPE (Projective Positional Encoding) support
    for camera-aware attention in HY-World/WorldPlay models.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        dtype: torch.dtype | None = None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
        prefix: str = "",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            supported_attention_backends=supported_attention_backends,
            prefix=prefix,
        )
        self.hidden_size = hidden_size

        # Add ProPE projection layer for camera-aware attention
        self.img_attn_prope_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            params_dtype=dtype,
            prefix=f"{prefix}.img_attn_prope_proj"
        )
        # Zero-initialize ProPE projection (starts as identity)
        nn.init.zeros_(self.img_attn_prope_proj.weight)
        if self.img_attn_prope_proj.bias is not None:
            nn.init.zeros_(self.img_attn_prope_proj.bias)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vec: torch.Tensor,
        vec_txt: torch.Tensor,
        freqs_cis: tuple,
        seq_attention_mask: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional ProPE camera conditioning.

        Args:
            img: Image/video tokens
            txt: Text tokens
            encoder_attention_mask: Text attention mask
            vec: Modulation vector
            freqs_cis: Rotary embedding frequencies
            seq_attention_mask: Sequence attention mask
            viewmats: Optional camera view matrices for ProPE [B, T, 4, 4]
            Ks: Optional camera intrinsics for ProPE [B, T, 3, 3]

        Returns:
            Tuple of (img, txt) output tokens
        """
        # Process modulation vectors (inherited from parent)
        img_mod_outputs = self.img_mod(vec)
        (
            img_attn_shift,
            img_attn_scale,
            img_attn_gate,
            img_mlp_shift,
            img_mlp_scale,
            img_mlp_gate,
        ) = torch.chunk(img_mod_outputs, 6, dim=-1)

        txt_mod_outputs = self.txt_mod(vec_txt)
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

        # begin hyworld: add camera pose through prope
        img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
            img_q.permute(0, 2, 1, 3),
            img_k.permute(0, 2, 1, 3),
            img_v.permute(0, 2, 1, 3),
            viewmats=viewmats,
            Ks=Ks,
        )  # [batch, num_heads, seqlen, head_dim]
        img_q_prope = img_q_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]
        img_k_prope = img_k_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]
        img_v_prope = img_v_prope.permute(
            0, 2, 1, 3
        )  # [batch, seqlen, num_heads, head_dim]
        # end hyworld

        from fastvideo.attention.backends.flash_attn import FlashAttnMetadataBuilder
        attn_metadata = FlashAttnMetadataBuilder().build(
            current_timestep=0,
            attn_mask=encoder_attention_mask,
        )
        # Run distributed attention
        with set_forward_context(current_timestep=0, attn_metadata=attn_metadata):
            img_attn, txt_attn = self.attn(img_q, img_k, img_v, txt_q, txt_k, txt_v, freqs_cis=freqs_cis, attention_mask=seq_attention_mask)
        
        # begin hyworld: attention with prope
        from fastvideo.attention.backends.flash_attn import FlashAttnMetadataBuilder
        attn_metadata_prope = FlashAttnMetadataBuilder().build(
            current_timestep=0,
            attn_mask=encoder_attention_mask,
        )
        with set_forward_context(current_timestep=0, attn_metadata=attn_metadata_prope):
            img_attn_prope, _ = self.attn(
                img_q_prope, img_k_prope, img_v_prope, txt_q, txt_k, txt_v, 
                freqs_cis=freqs_cis, attention_mask=seq_attention_mask
            )
            img_attn_prope = img_attn_prope.reshape(batch_size, image_seq_len, -1)
            img_attn_prope = rearrange(
                img_attn_prope, "B L (H D) -> B H L D", H=self.num_attention_heads
            )
            img_attn_prope = apply_fn_o(
                img_attn_prope
            )  # [batch, num_heads, seqlen, head_dim]
            img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
        # end hyworld: attention with prope

        # begin hyworld: add prope to img_attn
        img_attn_out, _ = self.img_attn_proj(img_attn.view(batch_size, image_seq_len, -1))
        img_attn_prope_out, _ = self.img_attn_prope_proj(img_attn_prope)
        img_attn_out = img_attn_out + img_attn_prope_out
        # end hyworld: add prope to img_attn

        # Use fused operation for residual connection, normalization, and modulation
        img_mlp_input, img_residual = self.img_attn_residual_mlp_norm(
            img, img_attn_out, img_attn_gate, img_mlp_shift, img_mlp_scale
        )

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


class HyWorldTransformer3DModel(HunyuanVideo15Transformer3DModel):
    r"""
    HY-World Transformer extending HunyuanVideo15 with:
    - ProPE (Projective Positional Encoding) for camera-aware attention
    - Action conditioning for interactive video generation
    """
    
    # Class attributes for weight loading - use HyWorld-specific mapping
    _fsdp_shard_conditions = HyWorldConfig().arch_config._fsdp_shard_conditions
    _compile_conditions = HyWorldConfig().arch_config._compile_conditions
    param_names_mapping = HyWorldConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = HyWorldConfig().arch_config.reverse_param_names_mapping

    def __init__(
        self,
        config: HyWorldConfig,
        hf_config: dict[str, Any],
    ) -> None:
        # Initialize parent class first
        # Note: We need to temporarily use HunyuanVideo15Config structure
        # The config should be compatible or we need to adapt it
        from fastvideo.configs.models.dits.hunyuanvideo15 import HunyuanVideo15Config
        
        # Create a compatible config for parent initialization
        # This assumes HyWorldConfig has similar structure to HunyuanVideo15Config
        super().__init__(
            config=HunyuanVideo15Config(
                arch_config=config.arch_config  # Share arch config
            ),
            hf_config=hf_config,
        )

        # Replace double_blocks with HY-World version that supports ProPE
        self.double_blocks = nn.ModuleList([
            HyWorldDoubleStreamBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                mlp_ratio=config.arch_config.mlp_ratio,
                dtype=None,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.double_blocks.{i}"
            )
            for i in range(config.arch_config.num_layers)
        ])

        # Add action conditioning module
        self.action_in = TimestepEmbedder(
            self.hidden_size,
            act_layer="silu",
            dtype=None,
            prefix=f"{config.prefix}.action_in"
        )
        # Zero-initialize action embedding (starts with no effect)
        nn.init.zeros_(self.action_in.mlp.fc_out.weight)
        if self.action_in.mlp.fc_out.bias is not None:
            nn.init.zeros_(self.action_in.mlp.fc_out.bias)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: list[torch.Tensor],
        encoder_attention_mask: list[torch.Tensor],
        action: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        timestep_r: Optional[torch.LongTensor] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        timestep_txt: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass with optional action and camera conditioning.

        Args:
            action: Optional action tensor for action conditioning [B, T] or [B*T]
            viewmats: Optional camera view matrices [B, T, 4, 4]
            Ks: Optional camera intrinsics [B, T, 3, 3]
            ... (other args same as parent)
        """
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
            (post_patch_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            self.config.rope_axes_dim,
            self.config.rope_theta
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # 2. Conditional embeddings
        temb = self.time_in(timestep, timestep_r=timestep_r)
        temb_txt = self.time_in(timestep_txt, timestep_r=timestep_r)

        # Add action conditioning if provided
        # temb shape: [B*T, C] where T = num_frames
        temb = temb + self.action_in(action.reshape(-1))
        
        # Keep a per-batch version for FinalLayer (average over frames)
        # [B*T, C] -> [B, T, C] -> [B, C]
        temb_final = temb.view(batch_size, -1, temb.shape[-1]).mean(dim=1)
        
        # Broadcast timestep embedding for transformer blocks (one per spatial token)
        # [B*T, C] -> [B, T*H*W, C] -> [B*T*H*W, C]
        temb = repeat(temb, "(B T) C -> B (T H W) C", B=batch_size, H=post_patch_height, W=post_patch_width)
        temb = rearrange(temb, "B S C -> (B S) C")

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

        # Prepare camera parameters for ProPE if provided
        viewmats_seq = repeat(
            viewmats, "B T M N->B (T H W) M N",
            H=post_patch_height,
            W=post_patch_width
        )
        Ks_seq = repeat(
            Ks, "B T M N->B (T H W) M N",
            H=post_patch_height,
            W=post_patch_width
        )

        # qwen text embedding
        encoder_hidden_states = self.txt_in(encoder_hidden_states, timestep_txt, encoder_attention_mask)

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

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.double_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    temb,
                    temb_txt,
                    freqs_cis,
                    seq_attention_mask,
                    viewmats_seq, # hyworld
                    Ks_seq, # hyworld
                )
        else:
            for block in self.double_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    temb,
                    temb_txt,
                    freqs_cis,
                    seq_attention_mask,
                    viewmats=viewmats_seq, # hyworld
                    Ks=Ks_seq, # hyworld
                )

        # Final layer processing
        # Use temb_final [B, C] instead of broadcasted temb [B*S, C] to avoid broadcast explosion
        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states = self.final_layer(hidden_states, temb_final)
        # Unpatchify to get original shape
        hidden_states = unpatchify(hidden_states, post_patch_num_frames, post_patch_height, post_patch_width, self.patch_size, self.out_channels)

        return hidden_states
