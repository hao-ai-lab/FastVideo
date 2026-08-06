# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 SandAI. All Rights Reserved.
"""Strict source-checkpoint loading for the MAGI-2 Wan image encoder."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastvideo.configs.models.vaes.magi2_wanvae import Magi2WanVAEConfig


CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """Apply left-padded temporal convolution with an optional feature cache."""

    def __init__(self, *args, **kwargs):
        """Store causal padding separately from the underlying convolution."""
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    @torch.compile
    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        """Convolve one temporal chunk after prepending the prior feature cache."""
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMSNorm(nn.Module):
    """Normalize channel vectors with the Wan encoder's root-mean-square rule."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        """Create the broadcastable scale and optional bias parameters."""
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the channel dimension and apply the learned scale."""
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


class Resample(nn.Module):
    """Downsample spatial dimensions and selected temporal transitions."""

    def __init__(self, dim: int, mode: str):
        """Build the spatial convolution and optional temporal convolution."""
        if mode not in ("downsample2d", "downsample3d"):
            raise ValueError(f"Unsupported MAGI-2 Wan encoder resampling mode: {mode}")
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=(2, 2)),
        )
        if mode == "downsample3d":
            self.time_conv = CausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
            )

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] = [0],  # noqa: B006 - compiled nested calls share this cache cursor.
    ) -> torch.Tensor:
        """Downsample one chunk and update its temporal feature cache."""
        batch_size, channels, time, height, width = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size, t=time)

        if self.mode == "downsample3d" and feat_cache is not None:
            index = feat_idx[0]
            if feat_cache[index] is None:
                feat_cache[index] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[index][:, :, -1:, :, :], x], 2))
                feat_cache[index] = cache_x
                feat_idx[0] += 1

        return x


class ResidualBlock(nn.Module):
    """Apply two cached causal convolutions and an additive shortcut."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        """Build the normalization, convolution, and shortcut modules."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] = [0],  # noqa: B006 - compiled nested calls share this cache cursor.
    ) -> torch.Tensor:
        """Run the residual path while preserving the causal convolution caches."""
        shortcut = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                index = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[index] is not None:
                    cache_x = torch.cat(
                        [feat_cache[index][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[index])
                feat_cache[index] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + shortcut


class AttentionBlock(nn.Module):
    """Apply single-head spatial self-attention independently per frame."""

    def __init__(self, dim: int):
        """Build normalization, fused query-key-value projection, and output projection."""
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the spatial positions of each frame and add the residual."""
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        query, key, value = (
            self.to_qkv(x)
            .reshape(batch_size * time, 1, channels * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(query, key, value)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=time)
        return x + identity


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Move each spatial patch into the channel dimension."""
    if patch_size == 1:
        return x
    if x.dim() == 4:
        return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    if x.dim() == 5:
        return rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=patch_size, r=patch_size)
    raise ValueError(f"Invalid MAGI-2 Wan image shape: {tuple(x.shape)}")


class AvgDown3D(nn.Module):
    """Average channel groups after spatial and temporal rearrangement."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        """Record the factors and channel-group size for the shortcut path."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        if in_channels * self.factor % out_channels != 0:
            raise ValueError("MAGI-2 Wan shortcut channels must divide evenly")
        self.group_size = in_channels * self.factor // out_channels

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange a video into channel groups and average each group."""
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        batch_size, channels, time, height, width = x.shape
        x = x.view(
            batch_size,
            channels,
            time // self.factor_t,
            self.factor_t,
            height // self.factor_s,
            self.factor_s,
            width // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            batch_size,
            channels * self.factor,
            time // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        x = x.view(
            batch_size,
            self.out_channels,
            self.group_size,
            time // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        return x.mean(dim=2)


class DownResidualBlock(nn.Module):
    """Combine residual encoder blocks with an averaged downsample shortcut."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temporal_downsample: bool = False,
        down_flag: bool = False,
    ):
        """Build the main and shortcut paths for one encoder resolution."""
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        downsamples: list[nn.Module] = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] = [0],  # noqa: B006 - compiled nested calls share this cache cursor.
    ) -> torch.Tensor:
        """Run one resolution block and combine its two paths."""
        shortcut_input = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(shortcut_input)


class Encoder3d(nn.Module):
    """Encode patchified conditioning images into Gaussian moments."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        attn_scales: tuple[float, ...],
        temporal_downsample: tuple[bool, ...],
        dropout: float,
    ):
        """Build the published MAGI-2 Wan image-encoder hierarchy."""
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample

        dims = [dim * multiplier for multiplier in [1, *dim_mult]]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downsamples: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            temporal_downsample_flag = temporal_downsample[index] if index < len(temporal_downsample) else False
            downsamples.append(
                DownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temporal_downsample=temporal_downsample_flag,
                    down_flag=index != len(dim_mult) - 1,
                )
            )
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] = [0],  # noqa: B006 - compiled nested calls share this cache cursor.
    ) -> torch.Tensor:
        """Encode one temporal chunk while updating every causal cache slot."""
        if feat_cache is not None:
            index = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[index] is not None:
                cache_x = torch.cat(
                    [feat_cache[index][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[index])
            feat_cache[index] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                index = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[index] is not None:
                    cache_x = torch.cat(
                        [feat_cache[index][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[index])
                feat_cache[index] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_causal_convolutions(model: nn.Module) -> int:
    """Count cache slots required by the encoder's causal convolutions."""
    return sum(isinstance(module, CausalConv3d) for module in model.modules())


class Magi2WanImageEncoder(nn.Module):
    """Encode MAGI-2 I2V reference images into normalized 48-channel latents."""

    def __init__(self, config: Magi2WanVAEConfig):
        """Build the encoder and quantization convolution with source checkpoint names."""
        super().__init__()
        self.z_dim = config.z_dim
        self.patch_size = config.patch_size
        self.encoder = Encoder3d(
            dim=config.base_dim,
            z_dim=config.z_dim * 2,
            dim_mult=tuple(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            attn_scales=tuple(config.attn_scales),
            temporal_downsample=tuple(config.temperal_downsample),
            dropout=config.dropout,
        )
        self.conv1 = CausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.register_buffer("mean", torch.tensor(config.latents_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("std", torch.tensor(config.latents_std, dtype=torch.float32), persistent=False)
        self.register_buffer("inverse_std", 1.0 / self.std, persistent=False)
        self.clear_cache()

    def clear_cache(self) -> None:
        """Reset every causal feature cache before and after an image batch."""
        self._enc_conv_num = count_causal_convolutions(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[torch.Tensor | None] = [None] * self._enc_conv_num

    def set_normalization(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        """Create normalization tensors on the encoder device with FP32 arithmetic."""
        device = self.conv1.weight.device
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.inverse_std = 1.0 / self.std

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode FP32 video produced by the pipeline's BF16 image round trip."""
        if video.dtype != torch.float32:
            raise TypeError(f"MAGI-2 Wan encoder input must be torch.float32, received {video.dtype}")
        self.clear_cache()
        video = patchify(video, patch_size=self.patch_size)
        time = video.shape[2]
        chunk_count = 1 + (time - 1) // 4
        for chunk_index in range(chunk_count):
            self._enc_conv_idx = [0]
            if chunk_index == 0:
                encoded = self.encoder(
                    video[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                encoded_chunk = self.encoder(
                    video[:, :, 1 + 4 * (chunk_index - 1) : 1 + 4 * chunk_index, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                encoded = torch.cat([encoded, encoded_chunk], 2)
        latent_mean, _ = self.conv1(encoded).chunk(2, dim=1)
        latent = (latent_mean - self.mean.view(1, self.z_dim, 1, 1, 1)) * self.inverse_std.view(
            1,
            self.z_dim,
            1,
            1,
            1,
        )
        self.clear_cache()
        return latent.float()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode one pipeline-ready reference-image tensor."""
        return self.encode(video)


def _is_decoder_tensor(source_name: str) -> bool:
    """Identify checkpoint tensors that belong only to video decoding."""
    return source_name.startswith("decoder.") or source_name.startswith("conv2.")


def load_magi2_wan_image_encoder(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> Magi2WanImageEncoder:
    """Build and strictly load every MAGI-2 Wan encoder and quant tensor."""
    config = Magi2WanVAEConfig()
    with torch.device("meta"):
        model = Magi2WanImageEncoder(config)
    target_state = model.state_dict()
    source_state = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    encoder_state: dict[str, torch.Tensor] = {}
    for source_name, source_tensor in source_state.items():
        if _is_decoder_tensor(source_name):
            continue
        if source_name not in target_state:
            raise KeyError(f"Unexpected MAGI-2 Wan encoder checkpoint tensor: {source_name}")
        target_tensor = target_state[source_name]
        if source_tensor.shape != target_tensor.shape or source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"MAGI-2 Wan encoder tensor metadata differs for {source_name}: "
                f"source={tuple(source_tensor.shape)}/{source_tensor.dtype}, "
                f"target={tuple(target_tensor.shape)}/{target_tensor.dtype}"
            )
        encoder_state[source_name] = source_tensor
    missing_names = sorted(set(target_state) - set(encoder_state))
    if missing_names:
        raise RuntimeError(f"MAGI-2 Wan encoder checkpoint is missing tensors: {missing_names}")
    incompatible = model.load_state_dict(encoder_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "MAGI-2 Wan encoder strict load failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.mean = torch.empty(len(config.latents_mean), dtype=torch.float32)
    model.std = torch.empty(len(config.latents_std), dtype=torch.float32)
    model.inverse_std = torch.empty(len(config.latents_std), dtype=torch.float32)
    model.to(device=device, dtype=torch.float32)
    model.set_normalization(config.latents_mean, config.latents_std)
    model.requires_grad_(False)
    return model.eval()


__all__ = ["Magi2WanImageEncoder", "load_magi2_wan_image_encoder"]
