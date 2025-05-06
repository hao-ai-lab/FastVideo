from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm

from fastvideo.v1.attention import DistributedAttention
from fastvideo.v1.configs.models.dits import FluxImageConfig
from fastvideo.v1.layers.layernorm import (LayerNormScaleShift, ScaleResidual,
                                           ScaleResidualLayerNormScaleShift)
from fastvideo.v1.layers.linear import ReplicatedLinear
from fastvideo.v1.layers.mlp import MLP
from fastvideo.v1.layers.rotary_embedding import (_apply_rotary_emb,
                                                  get_rotary_pos_embed)
from fastvideo.v1.layers.visual_embedding import (ModulateProjection,
                                                  TimestepEmbedder)
from fastvideo.v1.models.dits.base import BaseDiT
from fastvideo.v1.platforms import _Backend


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal DiT block with separate modulation for text and image/video,
    using distributed attention and linear layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float = 4.0,
        dtype: Optional[torch.dtype] = None,
        supported_attention_backends: Optional[Tuple[_Backend, ...]] = None,
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

        self.img_attn_q_norm = RMSNorm(head_dim, eps=1e-6)
        self.img_attn_k_norm = RMSNorm(head_dim, eps=1e-6)

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
        self.txt_attn_q_norm = RMSNorm(head_dim, eps=1e-6)
        self.txt_attn_k_norm = RMSNorm(head_dim, eps=1e-6)

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
        freqs_cis_img: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Apply rotary embeddings for image
        cos, sin = freqs_cis_img
        img_q, img_k = _apply_rotary_emb(
            img_q, cos, sin,
            is_neox_style=False), _apply_rotary_emb(img_k,
                                                    cos,
                                                    sin,
                                                    is_neox_style=False)
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

        # Run distributed attention
        img_attn, txt_attn = self.attn(img_q, img_k, img_v, txt_q, txt_k, txt_v)
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


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers using distributed attention
    and tensor parallelism.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float = 4.0,
        dtype: Optional[torch.dtype] = None,
        supported_attention_backends: Optional[Tuple[_Backend, ...]] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        head_dim = hidden_size // num_attention_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        # Combined QKV and MLP input projection
        self.linear1 = ReplicatedLinear(hidden_size,
                                        hidden_size * 3 + mlp_hidden_dim,
                                        bias=True,
                                        params_dtype=dtype,
                                        prefix=f"{prefix}.linear1")

        # Combined projection and MLP output
        self.linear2 = ReplicatedLinear(hidden_size + mlp_hidden_dim,
                                        hidden_size,
                                        bias=True,
                                        params_dtype=dtype,
                                        prefix=f"{prefix}.linear2")

        # QK norm layers
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        # Fused operations with better naming
        self.input_norm_scale_shift = LayerNormScaleShift(
            hidden_size,
            norm_type="layer",
            eps=1e-6,
            elementwise_affine=False,
            dtype=dtype)
        self.output_residual = ScaleResidual()

        # Activation function
        self.mlp_act = nn.GELU(approximate="tanh")

        # Modulation
        self.modulation = ModulateProjection(hidden_size,
                                             factor=3,
                                             act_layer="silu",
                                             dtype=dtype,
                                             prefix=f"{prefix}.modulation")

        # Distributed attention
        self.attn = DistributedAttention(
            num_heads=num_attention_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn")

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis_img: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Process modulation
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)

        # Apply pre-norm and modulation using fused operation
        x_mod = self.input_norm_scale_shift(x, mod_shift, mod_scale)

        # Get combined projections
        linear1_out, _ = self.linear1(x_mod)

        # Split into QKV and MLP parts
        qkv, mlp = torch.split(linear1_out,
                               [3 * self.hidden_size, self.mlp_hidden_dim],
                               dim=-1)

        # Process QKV
        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_attention_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply QK-Norm
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # Split into image and text parts
        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
        img_v, txt_v = v[:, :-txt_len], v[:, -txt_len:]
        # Apply rotary embeddings to image parts
        cos, sin = freqs_cis_img
        img_q, img_k = _apply_rotary_emb(
            img_q, cos, sin,
            is_neox_style=False), _apply_rotary_emb(img_k,
                                                    cos,
                                                    sin,
                                                    is_neox_style=False)

        # Run distributed attention
        img_attn_output, txt_attn_output = self.attn(img_q, img_k, img_v, txt_q,
                                                     txt_k, txt_v)
        attn_output = torch.cat((img_attn_output, txt_attn_output),
                                dim=1).view(batch_size, seq_len, -1)
        # Process MLP activation
        mlp_output = self.mlp_act(mlp)

        # Combine attention and MLP outputs
        combined = torch.cat((attn_output, mlp_output), dim=-1)

        # Final projection
        output, _ = self.linear2(combined)

        # Apply residual connection with gating using fused operation
        return self.output_residual(x, output, mod_gate)


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

        output_dim = patch_size**3 * out_channels

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

    def forward(self, img, vec):
        scale, shift = self.adaLN_modulation(vec).chunk(2, dim=-1)
        img = self.norm_final(img) * (1.0 +
                                      scale.unsqueeze(1)) + shift.unsqueeze(1)
        img, _ = self.linear(img)
        return img


class FluxTransformer2DModel(BaseDiT):
    _fsdp_shard_conditions = FluxImageConfig()._fsdp_shard_conditions
    _supported_attention_backends = FluxImageConfig(
    )._supported_attention_backends
    _param_names_mapping = FluxImageConfig()._param_names_mapping

    def __init__(self, config: FluxImageConfig) -> None:
        super().__init__(config=config)

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.text_states_dim = config.joint_attention_dim
        self.text_states_dim_2 = config.pooled_projection_dim
        self.rope_dim_list = list(config.axes_dims_rope)
        self.rope_theta = config.rope_theta
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        self.img_in = ReplicatedLinear(config.in_channels,
                                       self.hidden_size,
                                       params_dtype=config.dtype,
                                       prefix=f"{config.prefix}.img_in")
        self.txt_in = ReplicatedLinear(self.text_states_dim,
                                       self.hidden_size,
                                       params_dtype=config.dtype,
                                       prefix=f"{config.prefix}.txt_in")
        self.time_in = TimestepEmbedder(self.hidden_size,
                                        act_layer="silu",
                                        dtype=config.dtype,
                                        prefix=f"{config.prefix}.time_in")
        self.txt2_in = MLP(self.text_states_dim_2,
                           self.hidden_size,
                           self.hidden_size,
                           act_type="silu",
                           dtype=config.dtype,
                           prefix=f"{config.prefix}.txt2_in")
        self.guidance_in = (TimestepEmbedder(
            self.hidden_size,
            act_layer="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.guidance_in")
                            if config.guidance_embeds else None)

        # Double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                self.hidden_size,
                self.num_attention_heads,
                dtype=config.dtype,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.double_blocks.{i}")
            for i in range(config.num_layers)
        ])

        # Single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                self.hidden_size,
                self.num_attention_heads,
                dtype=config.dtype,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.single_blocks.{i+config.num_layers}")
            for i in range(config.num_single_layers)
        ])

        self.final_layer = FinalLayer(config.hidden_size,
                                      self.patch_size,
                                      self.out_channels,
                                      dtype=config.dtype,
                                      prefix=f"{config.prefix}.final_layer")

        self.__post_init__()

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: Optional[Union[
                    torch.Tensor, List[torch.Tensor]]] = None,
                guidance=None,
                **kwargs):
        """
        Forward pass of the FluxTransformer2DModel.
        
        Args:
            hidden_states: Input image latents [B, N, C]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Diffusion timestep
            guidance: Guidance scale for CFG
            
        Returns:
            Tuple of (output)
        """
        h = kwargs.pop("height_latents") or None
        w = kwargs.pop("width_latents") or None
        assert h is not None and w is not None

        img = x = hidden_states

        # Match diffusers implementation by multiplying timestep by 1000
        t = timestep.to(img.dtype)

        # Split text embeddings - first token is global, rest are per-token
        txt = encoder_hidden_states[1]
        text_states_2 = encoder_hidden_states[0]

        # Get spatial dimensions
        # _, _, oh, ow = img.shape
        th, tw = (h // self.patch_size // 2, w // self.patch_size // 2)

        # Get rotary embeddings
        freqs_cos_img, freqs_sin_img = get_rotary_pos_embed(
            (1, th, tw),
            self.hidden_size,
            self.num_attention_heads,
            self.rope_dim_list,
            self.rope_theta,
            shard_dim=1,
        )
        freqs_cos_img = freqs_cos_img.to(img.device)
        freqs_sin_img = freqs_sin_img.to(img.device)
        freqs_cis_img = (freqs_cos_img, freqs_sin_img)

        # Prepare modulation vectors
        vec = self.time_in(t)

        # Add text modulation
        vec = vec + self.txt2_in(text_states_2)

        # Add guidance modulation
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # embed text and image
        img, _ = self.img_in(img)
        txt, _ = self.txt_in(txt)
        img_seq_len = img.shape[1]
        txt_seq_len = txt.shape[1]

        # Process through double stream blocks
        for index, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis_img]
            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks
        x = torch.cat((img, txt), 1)

        # Process through single stream blocks
        for index, block in enumerate(self.single_blocks):
            single_block_args = [x, vec, txt_seq_len, freqs_cis_img]
            x = block(*single_block_args)

        # Extract image features
        img = x[:, :img_seq_len, ...]

        # Final layer
        img = self.final_layer(img, vec)

        return img
