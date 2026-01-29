# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models.dits import HunyuanGameCraftConfig
from fastvideo.configs.sample.teacache import TeaCacheParams
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.layernorm import (LayerNormScaleShift, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (_apply_rotary_emb,
                                               get_rotary_pos_embed)
from fastvideo.layers.visual_embedding import (ModulateProjection, PatchEmbed,
                                               TimestepEmbedder, unpatchify)
from fastvideo.models.dits.base import CachableDiT
from fastvideo.models.utils import modulate
from fastvideo.platforms import AttentionBackendEnum


class HunyuanRMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class CameraNet(nn.Module):
    """
    Camera state encoding network adapted for FastVideo.
    
    Processes camera parameters into feature embeddings for video generation.
    """

    def __init__(
        self,
        in_channels: int,
        downscale_coef: int,
        out_channels: int,
        patch_size: list,
        hidden_size: int,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        
        start_channels = in_channels * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]
        self.input_channels = input_channels
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)
        
        self.encode_first = nn.Sequential(
            nn.Conv2d(input_channels[0], input_channels[1], kernel_size=1, stride=1, padding=0, dtype=dtype),
            nn.GroupNorm(2, input_channels[1], dtype=dtype),
            nn.ReLU(),
        )
        self._initialize_weights(self.encode_first)
        
        self.encode_second = nn.Sequential(
            nn.Conv2d(input_channels[1], input_channels[2], kernel_size=1, stride=1, padding=0, dtype=dtype),
            nn.GroupNorm(2, input_channels[2], dtype=dtype),
            nn.ReLU(),
        )
        self._initialize_weights(self.encode_second)
        
        self.final_proj = nn.Conv2d(input_channels[2], out_channels, kernel_size=1, dtype=dtype)
        self.zeros_init_linear(self.final_proj)
        
        self.scale = nn.Parameter(torch.ones(1))
        
        self.camera_in = PatchEmbed(patch_size, out_channels, hidden_size, dtype=dtype, prefix=f"{prefix}.camera_in")

    def zeros_init_linear(self, linear: nn.Module):
        if isinstance(linear, (nn.Linear, nn.Conv2d)):
            if hasattr(linear, "weight"):
                nn.init.zeros_(linear.weight)
            if hasattr(linear, "bias") and linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def _initialize_weights(self, block):
        for m in block:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2.0 / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compress_time(self, x, num_frames):
        """Temporal dimension compression using average pooling."""
        x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
        batch_size, frames, channels, height, width = x.shape
        x = rearrange(x, 'b f c h w -> (b h w) c f')
        
        if x.shape[-1] == 66 or x.shape[-1] == 34:
            x_len = x.shape[-1]
            x_clip1 = x[..., :x_len//2]
            x_clip1_first, x_clip1_rest = x_clip1[..., 0].unsqueeze(-1), x_clip1[..., 1:]
            x_clip1_rest = F.avg_pool1d(x_clip1_rest, kernel_size=2, stride=2)

            x_clip2 = x[..., x_len//2:x_len]
            x_clip2_first, x_clip2_rest = x_clip2[..., 0].unsqueeze(-1), x_clip2[..., 1:]
            x_clip2_rest = F.avg_pool1d(x_clip2_rest, kernel_size=2, stride=2)

            x = torch.cat([x_clip1_first, x_clip1_rest, x_clip2_first, x_clip2_rest], dim=-1)
        elif x.shape[-1] % 2 == 1:
            x_first, x_rest = x[..., 0], x[..., 1:]
            if x_rest.shape[-1] > 0:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)
            x = torch.cat([x_first[..., None], x_rest], dim=-1)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
            
        x = rearrange(x, '(b h w) c f -> (b f) c h w', b=batch_size, h=height, w=width)
        return x

    def forward(self, camera_states: torch.Tensor):
        """
        Forward pass: encodes camera states into feature embeddings.
        
        Args:
            camera_states: Camera state tensor [B, T, C, H, W]
            
        Returns:
            Encoded feature embeddings after patch embedding and scaling
        """
        batch_size, num_frames, channels, height, width = camera_states.shape
        camera_states = rearrange(camera_states, 'b f c h w -> (b f) c h w')
        camera_states = self.unshuffle(camera_states)
        camera_states = self.encode_first(camera_states)
        camera_states = self.compress_time(camera_states, num_frames=num_frames)
        
        num_frames = camera_states.shape[0] // batch_size
        camera_states = self.encode_second(camera_states)
        camera_states = self.compress_time(camera_states, num_frames=num_frames)
        camera_states = self.final_proj(camera_states)
        camera_states = rearrange(camera_states, "(b f) c h w -> b c f h w", b=batch_size)
        camera_states = self.camera_in(camera_states)
        
        return camera_states * self.scale


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

        # QK Normalization layers (critical for numerical alignment!)
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
        img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :, 2]

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        
        # Apply rotary embeddings
        cos, sin = freqs_cis
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
        txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :, 2]

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
        dtype: torch.dtype | None = None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...]
        | None = None,
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
        self.q_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.k_norm = HunyuanRMSNorm(head_dim, eps=1e-6, dtype=dtype)

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
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
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
        cos, sin = freqs_cis
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


class HunyuanGameCraftTransformer3DModel(CachableDiT):
    """
    HunyuanGameCraft Transformer backbone adapted for distributed training.
    
    This implementation extends HunyuanVideo with camera conditioning for
    interactive game video generation.
    
    Based on the architecture from:
    - HunyuanGameCraft: https://arxiv.org/abs/2506.17201
    - Flux.1: https://github.com/black-forest-labs/flux
    - MMDiT: http://arxiv.org/abs/2403.03206
    """

    _fsdp_shard_conditions = HunyuanGameCraftConfig()._fsdp_shard_conditions
    _compile_conditions = HunyuanGameCraftConfig()._compile_conditions
    _supported_attention_backends = HunyuanGameCraftConfig()._supported_attention_backends
    param_names_mapping = HunyuanGameCraftConfig().param_names_mapping
    reverse_param_names_mapping = HunyuanGameCraftConfig().reverse_param_names_mapping
    lora_param_names_mapping = HunyuanGameCraftConfig().lora_param_names_mapping

    def __init__(self, config: HunyuanGameCraftConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)

        self.patch_size = [
            config.patch_size_t, config.patch_size, config.patch_size
        ]
        self.in_channels = config.in_channels
        self.num_channels_latents = config.num_channels_latents
        self.out_channels = config.in_channels if config.out_channels is None else config.out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embeds = config.guidance_embeds
        self.rope_dim_list = list(config.rope_axes_dim)
        self.rope_theta = config.rope_theta
        self.text_states_dim = config.text_embed_dim
        self.text_states_dim_2 = config.pooled_projection_dim
        self.dtype = config.dtype

        # GameCraft specific configs
        self.text_projection = config.text_projection
        self.use_attention_mask = config.use_attention_mask
        self.camera_in_channels = config.camera_in_channels
        self.camera_down_coef = config.camera_down_coef
        self.multitask_mask_training_type = config.multitask_mask_training_type

        pe_dim = config.hidden_size // config.num_attention_heads
        if sum(config.rope_axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.rope_axes_dim} but expected positional dim {pe_dim}"
            )

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents

        # Image projection
        self.img_in = PatchEmbed(self.patch_size,
                                 self.in_channels,
                                 self.hidden_size,
                                 dtype=config.dtype,
                                 prefix=f"{config.prefix}.img_in",
                                 multitask_mask_training_type=self.multitask_mask_training_type)

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = MLP(self.text_states_dim,
                             self.hidden_size,
                             self.hidden_size,
                             act_type="silu",
                             dtype=config.dtype,
                             prefix=f"{config.prefix}.txt_in")
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(self.text_states_dim,
                                             config.hidden_size,
                                             config.num_attention_heads,
                                             depth=config.num_refiner_layers,
                                             dtype=config.dtype,
                                             prefix=f"{config.prefix}.txt_in")
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # Time modulation
        self.time_in = TimestepEmbedder(self.hidden_size,
                                        act_layer="silu",
                                        dtype=config.dtype,
                                        prefix=f"{config.prefix}.time_in")

        # Text modulation
        self.vector_in = MLP(self.text_states_dim_2,
                             self.hidden_size,
                             self.hidden_size,
                             act_type="silu",
                             dtype=config.dtype,
                             prefix=f"{config.prefix}.vector_in")

        # Guidance modulation
        self.guidance_in = (TimestepEmbedder(
            self.hidden_size,
            act_layer="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.guidance_in")
                            if self.guidance_embeds else None)

        # Camera network (always outputs to base out_channels, not multitask-expanded channels)
        self.camera_net = CameraNet(
            in_channels=self.camera_in_channels,
            out_channels=self.out_channels,
            downscale_coef=self.camera_down_coef,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            dtype=config.dtype,
            prefix=f"{config.prefix}.camera_net"
        )

        # Double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                config.hidden_size,
                config.num_attention_heads,
                mlp_ratio=config.mlp_ratio,
                dtype=config.dtype,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.double_blocks.{i}")
            for i in range(config.num_layers)
        ])

        # Single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                config.hidden_size,
                config.num_attention_heads,
                mlp_ratio=config.mlp_ratio,
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
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                guidance=None,
                camera_latents: torch.Tensor | None = None,
                **kwargs):
        """
        Forward pass of the HunyuanGameCraft model.
        
        Args:
            hidden_states: Input image/video latents [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Diffusion timestep
            guidance: Guidance scale for CFG
            camera_latents: Camera trajectory latents [B, T, C, H, W]
            
        Returns:
            Output tensor
        """
        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        enable_teacache = forward_batch is not None and forward_batch.enable_teacache

        if guidance is None:
            guidance = torch.tensor([6016.0],
                                    device=hidden_states.device,
                                    dtype=hidden_states.dtype)

        img = x = hidden_states
        t = timestep

        # Split text embeddings
        if isinstance(encoder_hidden_states, torch.Tensor):
            txt = encoder_hidden_states[:, 1:]
            text_states_2 = encoder_hidden_states[:, 0, :self.text_states_dim_2]
        else:
            txt = encoder_hidden_states[0]
            text_states_2 = encoder_hidden_states[1]

        # Get spatial dimensions
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Get rotary embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (tt * get_sp_world_size(), th, tw), self.hidden_size,
            self.num_attention_heads, self.rope_dim_list, self.rope_theta)
        freqs_cos = freqs_cos.to(x.device)
        freqs_sin = freqs_sin.to(x.device)
        
        # Prepare modulation vectors
        vec = self.time_in(t)

        # Add text modulation
        vec = vec + self.vector_in(text_states_2)

        # Add guidance modulation if needed
        if self.guidance_in and guidance is not None:
            vec = vec + self.guidance_in(guidance)
            
        # Embed image and text
        img = self.img_in(img)
        
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        
        # Process camera latents
        if camera_latents is not None:
            latent_len = x.shape[2]  # temporal dimension
            
            if latent_len == 18:
                camera_features = torch.cat([
                    self.camera_net(torch.zeros_like(camera_latents)),
                    self.camera_net(camera_latents)
                ], dim=1)
            elif latent_len == 9:
                camera_features = self.camera_net(camera_latents)
            elif latent_len == 10:
                camera_features = torch.cat([
                    self.camera_net(torch.zeros_like(camera_latents[:, 0:4, :, :, :])),
                    self.camera_net(camera_latents)
                ], dim=1)
            else:
                camera_features = self.camera_net(camera_latents)
                
            img = img + camera_features

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        should_skip_forward = self.should_skip_forward_for_cached_states(
            img=img, vec=vec)

        if should_skip_forward:
            img = self.retrieve_cached_states(img)
        else:
            if enable_teacache:
                original_img = img.clone()

            # Process through double stream blocks
            for index, block in enumerate(self.double_blocks):
                double_block_args = [img, txt, vec, freqs_cis]
                img, txt = block(*double_block_args)
                
            # Merge txt and img to pass through single stream blocks
            x = torch.cat((img, txt), 1)

            # Process through single stream blocks
            if len(self.single_blocks) > 0:
                for index, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        freqs_cis,
                    ]
                    x = block(*single_block_args)

            # Extract image features
            img = x[:, :img_seq_len, ...]

            if enable_teacache:
                self.maybe_cache_states(img, original_img)

        # Final layer processing
        img = self.final_layer(img, vec)
        
        # Unpatchify to get original shape
        img = unpatchify(img, tt, th, tw, self.patch_size, self.out_channels)

        return img

    def maybe_cache_states(self, hidden_states: torch.Tensor,
                           original_hidden_states: torch.Tensor) -> None:
        self.previous_residual = hidden_states - original_hidden_states

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None:
            return False
        current_timestep = forward_context.current_timestep
        enable_teacache = forward_batch.enable_teacache

        if not enable_teacache:
            return False
            
        # TeaCache not yet supported for GameCraft
        return False

    def retrieve_cached_states(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.previous_residual


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
        self.input_embedder = ReplicatedLinear(
            in_channels,
            hidden_size,
            bias=True,
            params_dtype=dtype,
            prefix=f"{prefix}.input_embedder")

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

    def forward(self, x, t):
        # Get timestep embeddings
        timestep_aware_representations = self.t_embedder(t)

        # Get context-aware representations
        context_aware_representations = torch.mean(x, dim=1)
        context_aware_representations = self.c_embedder(
            context_aware_representations)
        c = timestep_aware_representations + context_aware_representations
        
        # Project input
        x, _ = self.input_embedder(x)
        
        # Process through refiner blocks
        for block in self.refiner_blocks:
            x = block(x, c)
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
        from fastvideo.attention import LocalAttention
        self.attn = LocalAttention(
            num_heads=num_attention_heads,
            head_size=hidden_size // num_attention_heads,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA),
        )

    def forward(self, x, c):
        # Get modulation parameters
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=-1)
        
        # Self-attention
        norm_x = self.norm1(x)
        qkv, _ = self.self_attn_qkv(norm_x)

        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_attention_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Run scaled dot product attention
        attn_output = self.attn(q, k, v)  # [B, L, H, D]
        attn_output = attn_output.reshape(batch_size, seq_len, -1)  # [B, L, H*D]

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
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x, _ = self.linear(x)
        return x

