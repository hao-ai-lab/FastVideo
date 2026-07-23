# SPDX-License-Identifier: Apache-2.0
"""Sparse point tracks to a dense, latent-aligned conditioning map."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(
    ids: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Map integer identity labels to sinusoidal embeddings."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) *
        torch.arange(half, device=ids.device, dtype=torch.float32) /
        max(half, 1))
    angles = ids.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TrackEncoder(nn.Module):
    """Encode ``[B, T, N, 2]`` normalized tracks on the Wan latent grid."""

    def __init__(
        self,
        id_dim: int,
        track_channels: int,
        vae_spatial_compression: int = 8,
        vae_temporal_compression: int = 4,
        max_track_id: int = 100_000,
        zero_init: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if id_dim <= 0 or track_channels <= 0:
            raise ValueError("id_dim and track_channels must be positive")
        if vae_spatial_compression <= 0 or vae_temporal_compression <= 0:
            raise ValueError("VAE compression ratios must be positive")
        if max_track_id <= 0:
            raise ValueError("max_track_id must be positive")

        self.id_dim = int(id_dim)
        self.track_channels = int(track_channels)
        self.vae_spatial_compression = int(vae_spatial_compression)
        self.vae_temporal_compression = int(vae_temporal_compression)
        self.max_track_id = int(max_track_id)
        self.use_bias = bool(use_bias)

        # Bias-free convolutions keep empty cells exactly zero. Left-padding
        # below gives the same temporal mapping as the causal Wan VAE:
        # T_latent = (T_pixel - 1) // ratio + 1.
        self.temporal_conv = nn.Conv3d(
            self.id_dim,
            self.track_channels,
            kernel_size=(self.vae_temporal_compression, 1, 1),
            stride=(self.vae_temporal_compression, 1, 1),
            bias=self.use_bias,
        )
        self.proj = nn.Conv3d(
            self.track_channels,
            self.track_channels,
            kernel_size=1,
            bias=self.use_bias,
        )
        if self.temporal_conv.bias is not None:
            nn.init.zeros_(self.temporal_conv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        if zero_init:
            nn.init.zeros_(self.proj.weight)

    def sample_ids(
        self,
        batch_size: int,
        num_tracks: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample IDs once at a batch/session boundary."""
        if num_tracks > self.max_track_id:
            raise ValueError(
                f"num_tracks ({num_tracks}) exceeds max_track_id "
                f"({self.max_track_id}); unique IDs are unavailable")
        return torch.stack([
            torch.randperm(
                self.max_track_id,
                device=device,
                generator=generator,
            )[:num_tracks] for _ in range(batch_size)
        ])

    def forward(
        self,
        coords: torch.Tensor,
        visibility: torch.Tensor,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        track_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords.ndim != 4 or coords.shape[-1] != 2:
            raise ValueError(
                "track coords must have shape [B, T, N, 2], got "
                f"{tuple(coords.shape)}")
        if visibility.shape != coords.shape[:-1]:
            raise ValueError(
                "track visibility must have shape [B, T, N] matching "
                f"coords, got {tuple(visibility.shape)}")
        if latent_t <= 0 or latent_h <= 0 or latent_w <= 0:
            raise ValueError("latent dimensions must be positive")

        batch_size, _, num_tracks, _ = coords.shape
        device = coords.device
        if num_tracks > self.max_track_id:
            raise ValueError(
                f"num_tracks ({num_tracks}) exceeds max_track_id "
                f"({self.max_track_id})")
        if track_ids is None:
            # A deterministic fallback is required for repeated denoising and
            # causal chunks. Training samples random IDs once in its wrapper.
            track_ids = torch.arange(
                num_tracks,
                device=device,
            ).unsqueeze(0).expand(batch_size, num_tracks)
        elif track_ids.shape != (batch_size, num_tracks):
            raise ValueError(
                "track_ids must have shape [B, N], got "
                f"{tuple(track_ids.shape)}")
        else:
            track_ids = track_ids.to(device=device)

        identities = sinusoidal_embedding(track_ids, self.id_dim)
        coords_float = coords.float()
        x = (coords_float[..., 0].clamp(0.0, 1.0) *
             (latent_w - 1)).round().long()
        y = (coords_float[..., 1].clamp(0.0, 1.0) *
             (latent_h - 1)).round().long()
        visible = visibility.to(device=device, dtype=torch.float32)
        contribution = visible.unsqueeze(-1) * identities.unsqueeze(1)

        cell_index = y * latent_w + x
        dense = torch.zeros(
            (
                batch_size,
                coords.shape[1],
                latent_h * latent_w,
                self.id_dim,
            ),
            device=device,
            dtype=torch.float32,
        )
        dense.scatter_add_(
            2,
            cell_index.unsqueeze(-1).expand(-1, -1, -1, self.id_dim),
            contribution,
        )
        dense = dense.view(
            batch_size,
            coords.shape[1],
            latent_h,
            latent_w,
            self.id_dim,
        ).permute(0, 4, 1, 2, 3).contiguous()
        dense = dense.to(dtype=self.temporal_conv.weight.dtype)
        dense = F.pad(
            dense,
            (0, 0, 0, 0, self.vae_temporal_compression - 1, 0),
        )
        dense = self.temporal_conv(dense)
        if dense.shape[2:] != (latent_t, latent_h, latent_w):
            dense = F.interpolate(
                dense,
                size=(latent_t, latent_h, latent_w),
                mode="nearest",
            )
        return self.proj(dense)
