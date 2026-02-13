# SPDX-License-Identifier: Apache-2.0
"""
HunyuanGameCraft Transformer model for FastVideo.

Ported from official Hunyuan-GameCraft-1.0 implementation.
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastvideo.configs.models.dits.hunyuangamecraft import (
    HunyuanGameCraftArchConfig,
    HunyuanGameCraftConfig,
)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import ModulateProjection, PatchEmbed, TimestepEmbedder, unpatchify
from fastvideo.models.dits.base import CachableDiT
from fastvideo.models.dits.hunyuanvideo import (
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
    SingleTokenRefiner,
)


class GameCraftFinalLayer(nn.Module):
    """
    GameCraft-specific FinalLayer with correct shift/scale order.
    
    The official GameCraft implementation uses shift, scale order (not scale, shift).
    This differs from the HunyuanVideo FinalLayer.
    """

    def __init__(self,
                 hidden_size,
                 patch_size,
                 out_channels,
                 dtype=None,
                 prefix: str = "") -> None:
        super().__init__()

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

        self.adaLN_modulation = ModulateProjection(
            hidden_size,
            factor=2,
            act_layer="silu",
            dtype=dtype,
            prefix=f"{prefix}.adaLN_modulation")

    def forward(self, x, c):
        # GameCraft uses shift, scale order (verified against official implementation)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x, _ = self.linear(x)
        return x


class CameraNet(nn.Module):
    """
    Camera state encoding network - ported from official GameCraft.
    
    Processes camera parameters (Plücker coordinates) into feature embeddings.
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        downscale_coef: int = 8,
        out_channels: int = 16,
        patch_size: list[int] = [1, 2, 2],
        hidden_size: int = 3072,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        _ = prefix  # Unused
        
        start_channels = in_channels * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]
        self.input_channels = input_channels
        
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)
        
        self.encode_first = nn.Sequential(
            nn.Conv2d(input_channels[0], input_channels[1], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[1]),
            nn.ReLU(),
        )
        self._initialize_weights(self.encode_first)
        
        self.encode_second = nn.Sequential(
            nn.Conv2d(input_channels[1], input_channels[2], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[2]),
            nn.ReLU(),
        )
        self._initialize_weights(self.encode_second)
        
        self.final_proj = nn.Conv2d(input_channels[2], out_channels, kernel_size=1)
        self._zeros_init_linear(self.final_proj)
        
        self.scale = nn.Parameter(torch.ones(1))
        
        self.camera_in = PatchEmbed(
            patch_size=patch_size,
            in_chans=out_channels,
            embed_dim=hidden_size,
        )
    
    def _zeros_init_linear(self, linear):
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
    
    def compress_time(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
        batch_size, frames, channels, height, width = x.shape
        x = rearrange(x, 'b f c h w -> (b h w) c f')
        
        if x.shape[-1] == 66 or x.shape[-1] == 34:
            x_len = x.shape[-1]
            x_clip1 = x[..., :x_len // 2]
            x_clip1_first = x_clip1[..., 0].unsqueeze(-1)
            x_clip1_rest = F.avg_pool1d(x_clip1[..., 1:], kernel_size=2, stride=2)
            x_clip2 = x[..., x_len // 2:]
            x_clip2_first = x_clip2[..., 0].unsqueeze(-1)
            x_clip2_rest = F.avg_pool1d(x_clip2[..., 1:], kernel_size=2, stride=2)
            x = torch.cat([x_clip1_first, x_clip1_rest, x_clip2_first, x_clip2_rest], dim=-1)
        elif x.shape[-1] % 2 == 1:
            x_first = x[..., 0]
            x_rest = x[..., 1:]
            if x_rest.shape[-1] > 0:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)
            x = torch.cat([x_first[..., None], x_rest], dim=-1)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        
        x = rearrange(x, '(b h w) c f -> (b f) c h w', b=batch_size, h=height, w=width)
        return x
    
    def forward(self, camera_states: torch.Tensor) -> torch.Tensor:
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


class HunyuanGameCraftTransformer3DModel(CachableDiT):
    """
    HunyuanGameCraft Transformer - ported from official implementation.
    """
    
    _fsdp_shard_conditions = HunyuanGameCraftArchConfig()._fsdp_shard_conditions
    _compile_conditions = HunyuanGameCraftArchConfig()._compile_conditions
    _supported_attention_backends = HunyuanGameCraftArchConfig()._supported_attention_backends
    param_names_mapping = HunyuanGameCraftConfig().param_names_mapping
    reverse_param_names_mapping = HunyuanGameCraftConfig().reverse_param_names_mapping
    
    def __init__(self, config: HunyuanGameCraftConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)
        
        arch = config.arch_config
        
        if isinstance(arch.patch_size, (list, tuple)):
            self.patch_size = list(arch.patch_size)
        else:
            self.patch_size = [arch.patch_size_t, arch.patch_size, arch.patch_size]
        
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        self.unpatchify_channels = self.out_channels
        self.num_channels_latents = self.out_channels  # Alias for latent_preparation stage
        self.hidden_size = arch.hidden_size
        self.num_heads = arch.num_attention_heads
        self.num_attention_heads = arch.num_attention_heads  # Alias for compatibility
        self.guidance_embeds = arch.guidance_embeds
        self.rope_dim_list = list(arch.rope_axes_dim)
        self.rope_theta = arch.rope_theta
        self.text_states_dim = arch.text_embed_dim
        self.text_states_dim_2 = arch.pooled_projection_dim
        self.dtype = arch.dtype
        
        pe_dim = self.hidden_size // self.num_heads
        if sum(self.rope_dim_list) != pe_dim:
            raise ValueError(f"rope_axes_dim sum {sum(self.rope_dim_list)} != {pe_dim}")
        
        factory_kwargs = {'dtype': self.dtype}
        
        self.img_in = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
            **factory_kwargs,
        )
        
        self.txt_in = SingleTokenRefiner(
            self.text_states_dim,
            self.hidden_size,
            self.num_heads,
            depth=arch.num_refiner_layers,
            **factory_kwargs,
        )
        
        self.time_in = TimestepEmbedder(self.hidden_size, **factory_kwargs)
        
        self.vector_in = MLP(
            self.text_states_dim_2,
            self.hidden_size,
            self.hidden_size,
            act_type="silu",
            **factory_kwargs,
        )
        
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, **factory_kwargs)
            if self.guidance_embeds else None
        )
        
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_heads,
                mlp_ratio=arch.mlp_ratio,
                supported_attention_backends=self._supported_attention_backends,
                **factory_kwargs,
            )
            for _ in range(arch.num_layers)
        ])
        
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_heads,
                mlp_ratio=arch.mlp_ratio,
                supported_attention_backends=self._supported_attention_backends,
                **factory_kwargs,
            )
            for _ in range(arch.num_single_layers)
        ])
        
        self.final_layer = GameCraftFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            **factory_kwargs,
        )
        
        self.camera_net = CameraNet(
            in_channels=arch.camera_in_channels,
            out_channels=16,
            downscale_coef=arch.camera_downscale_coef,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        timestep: torch.Tensor,
        camera_states: torch.Tensor | None = None,
        encoder_attention_mask: list[torch.Tensor] | None = None,
        guidance: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        img = x
        _, _, ot, oh, ow = x.shape
        tt = ot // self.patch_size[0]
        th = oh // self.patch_size[1]
        tw = ow // self.patch_size[2]
        
        text_states = encoder_hidden_states[0]
        text_states_2 = encoder_hidden_states[1] if len(encoder_hidden_states) > 1 else None
        text_mask = encoder_attention_mask[0] if encoder_attention_mask else None
        
        vec = self.time_in(timestep)
        
        if text_states_2 is not None:
            vec = vec + self.vector_in(text_states_2)
        
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(guidance)
        
        img = self.img_in(img)
        
        if camera_states is not None:
            latent_len = ot
            if latent_len == 18:
                camera_latents = torch.cat([
                    self.camera_net(torch.zeros_like(camera_states)),
                    self.camera_net(camera_states)
                ], dim=1)
            elif latent_len == 9:
                camera_latents = self.camera_net(camera_states)
            elif latent_len == 10:
                camera_latents = torch.cat([
                    self.camera_net(torch.zeros_like(camera_states[:, 0:4, :, :, :])),
                    self.camera_net(camera_states)
                ], dim=1)
            else:
                camera_latents = self.camera_net(camera_states)
            img = img + camera_latents
        
        txt = self.txt_in(text_states, timestep)
        txt_seq_len = txt.shape[1]
        
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (tt, th, tw),
            self.hidden_size,
            self.num_heads,
            self.rope_dim_list,
            self.rope_theta,
        )
        freqs_cos = freqs_cos.to(device=img.device, dtype=img.dtype)
        freqs_sin = freqs_sin.to(device=img.device, dtype=img.dtype)
        freqs_cis = (freqs_cos, freqs_sin)
        
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, freqs_cis)
        
        x = torch.cat([img, txt], dim=1)
        
        for block in self.single_blocks:
            x = block(x, vec, txt_seq_len, freqs_cis)
        
        img = x[:, :-txt_seq_len, ...]
        img = self.final_layer(img, vec)
        img = unpatchify(img, tt, th, tw, self.patch_size, self.out_channels)
        
        return img
