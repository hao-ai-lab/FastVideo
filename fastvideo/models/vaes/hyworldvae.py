# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers and HY-WorldPlay

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

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from fastvideo.layers.activation import get_act_fn
from fastvideo.configs.models.vaes import Hunyuan15VAEConfig
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.models.vaes.hunyuan15vae import (
    HunyuanVideo15RMS_norm as HYWorldRMS_norm,
    HunyuanVideo15AttnBlock as HYWorldAttnBlock,
)

# Cache size for temporal feature caching (number of frames to cache)
CACHE_T = 2

class HYWorldCausalConv3d(nn.Module):
    """Causal Conv3d with optional cache support for temporal feature caching."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        # Padding format: (W_left, W_right, H_left, H_right, T_left, T_right)
        self.time_causal_padding = (
            kernel_size[0] // 2,  # W_left (spatial)
            kernel_size[0] // 2,  # W_right (spatial)
            kernel_size[1] // 2,  # H_left (spatial)
            kernel_size[1] // 2,  # H_right (spatial)
            kernel_size[2] - 1,   # T_left (temporal causal padding)
            0,                    # T_right (no future padding for causal)
        )

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, hidden_states: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional temporal caching.
        
        Args:
            hidden_states: Input tensor of shape (B, C, T, H, W)
            cache_x: Optional cached frames from previous chunk, shape (B, C, CACHE_T, H, W)
                    When provided, uses cached frames instead of padding for temporal dimension.
        """
        padding = list(self.time_causal_padding)
        
        if cache_x is not None and self.time_causal_padding[4] > 0:  # Has temporal padding and cache
            cache_x = cache_x.to(hidden_states.device)
            # Concatenate cached frames with current input on temporal dimension
            hidden_states = torch.cat([cache_x, hidden_states], dim=2)
            # Reduce temporal padding since we have cached frames
            padding[4] -= cache_x.shape[2]
        
        hidden_states = F.pad(hidden_states, padding, mode=self.pad_mode)
        return self.conv(hidden_states)


class HYWorldUpsample(nn.Module):
    """Hierarchical upsampling with temporal/spatial support and optional caching."""
    
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = HYWorldCausalConv3d(in_channels, out_channels * factor, kernel_size=3)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(
        self, 
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
        first_chunk: bool = False,
    ):
        """
        Forward pass with optional temporal caching.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
            first_chunk: Whether this is the first chunk (affects temporal upsample behavior)
        """
        r1 = 2 if self.add_temporal_upsample else 1
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            h = self.conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            h = self.conv(x)

        if self.add_temporal_upsample:
            if first_chunk:
                # First chunk: only spatial upsample
                h = rearrange(h, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
                h = h[:, : h.shape[1] // 2]
                # Compute the shortcut part
                shortcut = rearrange(x, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
                shortcut = shortcut.repeat_interleave(repeats=self.repeats // 2, dim=1)
            elif feat_cache is None and x.shape[2] > 1:
                # No cache and multiple frames: separate first frame and rest
                h_first = h[:, :, :1, :, :]
                h_rest = h[:, :, 1:, :, :]
                x_first = x[:, :, :1, :, :]
                x_rest = x[:, :, 1:, :, :]

                # First frame: only spatial upsample
                h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
                h_first = h_first[:, : h_first.shape[1] // 2]
                shortcut_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
                shortcut_first = shortcut_first.repeat_interleave(repeats=self.repeats // 2, dim=1)
                out_first = h_first + shortcut_first

                # Remaining frames: spatio-temporal upsample
                h_rest = rearrange(h_rest, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
                shortcut_rest = rearrange(x_rest, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
                shortcut_rest = shortcut_rest.repeat_interleave(repeats=self.repeats, dim=1)
                out_rest = h_rest + shortcut_rest

                return torch.cat([out_first, out_rest], dim=2)
            else:
                # Subsequent chunks with cache: spatio-temporal upsample
                h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
                shortcut = rearrange(x, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
                shortcut = shortcut.repeat_interleave(repeats=self.repeats, dim=1)
        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        
        return h + shortcut


class HYWorldDownsample(nn.Module):
    """Hierarchical downsampling with temporal/spatial support and optional caching."""
    
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        self.conv = HYWorldCausalConv3d(in_channels, out_channels // factor, kernel_size=3)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(
        self, 
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
    ):
        """
        Forward pass with optional temporal caching.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
        """
        r1 = 2 if self.add_temporal_downsample else 1
        
        # Apply conv with caching
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            h = self.conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            h = self.conv(x) # Change the channel, ready for spatial or temporal downsample
        
        if self.add_temporal_downsample:
            if x.shape[2] == 1:
                # Single frame: only spatial downsample
                h = rearrange(h, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
                h = torch.cat([h, h], dim=1)
                # Compute the shortcut part
                shortcut = rearrange(x, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
                B, C, T, H, W = shortcut.shape
                shortcut = shortcut.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)
            else:
                # Multiple frames: full spatio-temporal downsample
                h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
                # Shortcut computation
                shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
                B, C, T, H, W = shortcut.shape
                shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut


class HYWorldResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.nonlinearity = get_act_fn(non_linearity)

        self.norm1 = HYWorldRMS_norm(in_channels, images=False)
        self.conv1 = HYWorldCausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = HYWorldRMS_norm(out_channels, images=False)
        self.conv2 = HYWorldCausalConv3d(out_channels, out_channels, kernel_size=3)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional temporal feature caching.
        
        Args:
            hidden_states: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
        """
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        
        # apply the feature cacheing mechanism
        if feat_cache is not None and feat_idx is not None:
            # Retrieve the current layer index.
            idx = feat_idx[0]

            # Clone the last CACHE_T frames from the current input to store for the next step.
            cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
            
            # Handle boundary conditions: if the current temporal chunk is too short (< 2 frames)
            # and we have a previous cache, prepend the last frame of the previous cache.
            # This ensures sufficient temporal context for the convolution kernel.
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # actually this means chunk 1 after chunk 0
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            
            # Apply the convolution layer using `hidden_states`and the *previous* cached state.
            hidden_states = self.conv1(hidden_states, feat_cache[idx])
            
            # Update the cache for this layer with the newly prepared context ('cache_x').
            feat_cache[idx] = cache_x

            # Increment the global layer index.
            feat_idx[0] += 1
        else:
            hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        # Second conv with caching
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            hidden_states = self.conv2(hidden_states, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)

        return hidden_states + residual


class HYWorldMidBlock(nn.Module):
    """Mid block with attention and resnet blocks, with optional caching support."""
    
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        add_attention: bool = True,
    ) -> None:
        super().__init__()
        self.add_attention = add_attention

        # There is always at least one resnet
        resnets = [
            HYWorldResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(HYWorldAttnBlock(in_channels))
            else:
                attentions.append(None)

            resnets.append(
                HYWorldResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self, 
        hidden_states: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional temporal caching.
        
        Args:
            hidden_states: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
        """
        hidden_states = self.resnets[0](hidden_states, feat_cache, feat_idx)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, feat_cache, feat_idx)

        return hidden_states


class HYWorldDownBlock3D(nn.Module):
    """Down block with resnet blocks and optional downsampling, with caching support."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample_out_channels: Optional[int] = None,
        add_temporal_downsample: int = True,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HYWorldResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if downsample_out_channels is not None:
            self.downsamplers = nn.ModuleList(
                [
                    HYWorldDownsample(
                        out_channels,
                        out_channels=downsample_out_channels,
                        add_temporal_downsample=add_temporal_downsample,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self, 
        hidden_states: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional temporal caching.
        
        Args:
            hidden_states: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
        """
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, feat_cache, feat_idx)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, feat_cache, feat_idx)

        return hidden_states


class HYWorldUpBlock3D(nn.Module):
    """Up block with resnet blocks and optional upsampling, with caching support."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample_out_channels: Optional[int] = None,
        add_temporal_upsample: bool = True,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HYWorldResnetBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if upsample_out_channels is not None:
            self.upsamplers = nn.ModuleList(
                [
                    HYWorldUpsample(
                        out_channels,
                        out_channels=upsample_out_channels,
                        add_temporal_upsample=add_temporal_upsample,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self, 
        hidden_states: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional temporal caching.
        
        Args:
            hidden_states: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
            first_chunk: Whether this is the first chunk (for upsampling behavior)
        """
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for resnet in self.resnets:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states, feat_cache, feat_idx)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, feat_cache, feat_idx, first_chunk)

        return hidden_states


class HYWorldEncoder3D(nn.Module):
    r"""
    3D vae encoder for HunyuanImageRefiner with optional temporal caching.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024, 1024),
        layers_per_block: int = 2,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 16,
        downsample_match_channel: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = block_out_channels[-1] // self.out_channels
        self.block_out_channels = block_out_channels

        self.conv_in = HYWorldCausalConv3d(in_channels, block_out_channels[0], kernel_size=3)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            add_spatial_downsample = i < np.log2(spatial_compression_ratio)
            output_channel = block_out_channels[i]
            if not add_spatial_downsample:
                down_block = HYWorldDownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=None,
                    add_temporal_downsample=False,
                )
                input_channel = output_channel
            else:
                add_temporal_downsample = i >= np.log2(spatial_compression_ratio // temporal_compression_ratio)
                downsample_out_channels = block_out_channels[i + 1] if downsample_match_channel else output_channel
                down_block = HYWorldDownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=downsample_out_channels,
                    add_temporal_downsample=add_temporal_downsample,
                )
                input_channel = downsample_out_channels

            self.down_blocks.append(down_block)

        self.mid_block = HYWorldMidBlock(in_channels=block_out_channels[-1])

        self.norm_out = HYWorldRMS_norm(block_out_channels[-1], images=False)
        self.conv_act = nn.SiLU()
        self.conv_out = HYWorldCausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(
        self, 
        hidden_states: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional temporal caching.
        
        Args:
            hidden_states: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
        """
        # conv_in with caching
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            hidden_states = self.conv_in(hidden_states, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(down_block, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states, feat_cache, feat_idx)
            hidden_states = self.mid_block(hidden_states, feat_cache, feat_idx)

        batch_size, _, frame, height, width = hidden_states.shape
        short_cut = hidden_states.view(batch_size, -1, self.group_size, frame, height, width).mean(dim=2)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)

        # conv_out with caching
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            hidden_states = self.conv_out(hidden_states, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            hidden_states = self.conv_out(hidden_states)

        hidden_states += short_cut

        return hidden_states


class HYWorldDecoder3D(nn.Module):
    r"""
    Causal decoder for 3D video-like data used for HunyuanImage-1.5 Refiner with optional temporal caching.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128),
        layers_per_block: int = 2,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.repeat = block_out_channels[0] // self.in_channels

        self.conv_in = HYWorldCausalConv3d(self.in_channels, block_out_channels[0], kernel_size=3)
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = HYWorldMidBlock(in_channels=block_out_channels[0])

        # up
        input_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            output_channel = block_out_channels[i]

            add_spatial_upsample = i < np.log2(spatial_compression_ratio)
            add_temporal_upsample = i < np.log2(temporal_compression_ratio)
            if add_spatial_upsample or add_temporal_upsample:
                upsample_out_channels = block_out_channels[i + 1] if upsample_match_channel else output_channel
                up_block = HYWorldUpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=upsample_out_channels,
                    add_temporal_upsample=add_temporal_upsample,
                )
                input_channel = upsample_out_channels
            else:
                up_block = HYWorldUpBlock3D(
                    num_layers=self.layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=None,
                    add_temporal_upsample=False,
                )
                input_channel = output_channel

            self.up_blocks.append(up_block)

        # out
        self.norm_out = HYWorldRMS_norm(block_out_channels[-1], images=False)
        self.conv_act = nn.SiLU()
        self.conv_out = HYWorldCausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(
        self, 
        hidden_states: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional temporal caching.
        
        Args:
            hidden_states: Input tensor of shape (B, C, T, H, W)
            feat_cache: List of cached features for each conv layer
            feat_idx: List containing current cache index [idx]
            first_chunk: Whether this is the first chunk (for upsampling behavior)
        """
        # conv_in with caching
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            hidden_states = self.conv_in(hidden_states, feat_cache[idx]) + hidden_states.repeat_interleave(
                repeats=self.repeat, dim=1
            )
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            hidden_states = self.conv_in(hidden_states) + hidden_states.repeat_interleave(repeats=self.repeat, dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(up_block, hidden_states)
        else:
            hidden_states = self.mid_block(hidden_states, feat_cache, feat_idx)
            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, feat_cache, feat_idx, first_chunk)

        # post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)

        # conv_out with caching
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = hidden_states[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            hidden_states = self.conv_out(hidden_states, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            hidden_states = self.conv_out(hidden_states)
        
        return hidden_states


class AutoencoderKLHYWorld(nn.Module, ParallelTiledVAE):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Revised from HunyuanVideo-1.5 with temporal caching support for HY-WorldPlay integration.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Hunyuan15VAEConfig, # HYWorld use the same VAE architecture as HunyuanVideo-1.5
    ) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)

        self.encoder: Optional[HYWorldEncoder3D] = None
        self.decoder: Optional[HYWorldDecoder3D] = None

        if config.load_encoder:
            self.encoder = HYWorldEncoder3D(
                in_channels=config.in_channels,
                out_channels=config.latent_channels * 2,
                block_out_channels=config.block_out_channels,
                layers_per_block=config.layers_per_block,
                temporal_compression_ratio=config.temporal_compression_ratio,
                spatial_compression_ratio=config.spatial_compression_ratio,
                downsample_match_channel=config.downsample_match_channel,
            )

        if config.load_decoder:
            self.decoder = HYWorldDecoder3D(
                in_channels=config.latent_channels,
                out_channels=config.out_channels,
                block_out_channels=list(reversed(config.block_out_channels)),
                layers_per_block=config.layers_per_block,
                temporal_compression_ratio=config.temporal_compression_ratio,
                spatial_compression_ratio=config.spatial_compression_ratio,
                upsample_match_channel=config.upsample_match_channel,
            )

        # TODO: Add spatial tiling.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 2000 # Fill in a random large number, as hy1.5 vae does not use temporal tiling

        # Cache-related attributes (initialized in clear_cache)
        self._conv_num: int = 0
        self._conv_idx: List[int] = [0]
        self._feat_map: List[Optional[torch.Tensor]] = []
        self._enc_conv_num: int = 0
        self._enc_conv_idx: List[int] = [0]
        self._enc_feat_map: List[Optional[torch.Tensor]] = []

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": (
                sum(1 for m in self.decoder.modules() if isinstance(m, HYWorldCausalConv3d))
                if self.decoder is not None
                else 0
            ),
            "encoder": (
                sum(1 for m in self.encoder.modules() if isinstance(m, HYWorldCausalConv3d))
                if self.encoder is not None
                else 0
            ),
        }

    def clear_cache(self) -> None:
        """
        Initialize/clear the feature cache for chunk-based encoding/decoding.
        
        This should be called before starting a new encode/decode sequence to ensure
        the cache is properly initialized.
        """
        # Cache for decoder
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map: List[Optional[torch.Tensor]] = [None] * self._conv_num
        
        # Cache for encoder
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map: List[Optional[torch.Tensor]] = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video with temporal caching for chunk-based processing.
        
        This processes video in chunks (first frame separately, then 4 frames at a time)
        while maintaining temporal context through caching. This matches the HY-WorldPlay
        behavior for memory-efficient long video encoding.
        
        Args:
            x: Input video tensor of shape (B, C, T, H, W)
            
        Returns:
            Encoded latent tensor
        """
        assert self.encoder is not None, "Encoder not loaded"
        _, _, num_frame, _, _ = x.shape

        self.clear_cache()
        
        # Process in chunks: first frame alone, then groups of 4 frames
        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                # First frame
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                # Subsequent frames in groups of 4
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], dim=2)

        self.clear_cache()
        return out

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents with temporal caching for chunk-based processing.
        
        This processes latents one frame at a time while maintaining temporal context
        through caching. This matches the HY-WorldPlay behavior for memory-efficient
        long video decoding.
        
        Args:
            z: Latent tensor of shape (B, C, T, H, W)
            
        Returns:
            Decoded video tensor
        """
        assert self.decoder is not None, "Decoder not loaded"
        _, _, num_frame, _, _ = z.shape

        self.clear_cache()
        
        # Process one frame at a time with caching
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                # First frame with first_chunk=True
                out = self.decoder(
                    z[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                # Subsequent frames
                out_ = self.decoder(
                    z[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=False,
                )
                out = torch.cat([out, out_], dim=2)

        self.clear_cache()
        return out

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                Generator for sampling.
        """
        x = sample

        # encode() uses temporal caching by default (via _encode)
        posterior = self.encode(x)

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # decode() uses temporal caching by default (via _decode)
        dec = self.decode(z)

        return dec

# Entry point for model registry
EntryClass = AutoencoderKLHYWorld
