# SPDX-License-Identifier: Apache-2.0
"""MotionStream-style point-track conditioning encoder.

Turns sparse point tracks into a dense, latent-aligned conditioning tensor that
``TrackWanTransformer3DModel`` channel-concatenates onto the video latent before patchifying.

This module is the **shared train/inference seam**: the same code that embeds CoTracker tracks
during training embeds the user's dragged points during real-time inference, so there is no
train/inference skew.

Representation (MotionStream, arXiv 2511.01266, eq. for c_m):
  - Each track ``n`` gets a ``d``-dim embedding ``phi_n`` from a randomly sampled integer ID via
    sinusoidal positional encoding. The ID is random per-sample at train time (so the model treats
    the embedding as an arbitrary identity tag instead of memorizing it) and assigned per dragged
    trajectory at inference time.
  - Build ``c_m[t, floor(y/s), floor(x/s)] = visibility[t, n] * phi_n`` on the latent spatial grid
    (``s`` = VAE spatial compression). Occluded / unspecified tracks contribute 0.
  - A small "track head" (4x temporal-compression conv + 1x1x1 conv) aligns the per-frame map to
    the temporally-compressed latent and projects ``d -> track_channels``.

Augmentations (1-200 point subsampling, p=0.2 temporal masking, text/motion CFG dropout) live in
the dataloader / training method, NOT here. This module just consumes whatever (coords, visibility)
it is handed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(ids: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Map integer/float ids ``(...)`` to sinusoidal embeddings ``(..., dim)`` (float32)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=ids.device, dtype=torch.float32) / max(half, 1))
    args = ids.float().unsqueeze(-1) * freqs  # (..., half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb  # (..., dim)


class TrackEncoder(nn.Module):
    """Sparse tracks -> dense latent-aligned conditioning map ``(B, track_channels, T_lat, H_lat, W_lat)``."""

    def __init__(self,
                 id_dim: int,
                 track_channels: int,
                 vae_spatial_compression: int = 8,
                 vae_temporal_compression: int = 4,
                 max_track_id: int = 100_000,
                 zero_init: bool = True) -> None:
        super().__init__()
        self.id_dim = id_dim
        self.track_channels = track_channels
        self.vae_spatial_compression = vae_spatial_compression
        self.vae_temporal_compression = vae_temporal_compression
        self.max_track_id = max_track_id

        # Track head: 4x temporal compression followed by a 1x1x1 conv (MotionStream).
        self.temporal_conv = nn.Conv3d(id_dim,
                                       track_channels,
                                       kernel_size=(vae_temporal_compression, 1, 1),
                                       stride=(vae_temporal_compression, 1, 1))
        self.proj = nn.Conv3d(track_channels, track_channels, kernel_size=1)
        if zero_init:
            # Start with zero track contribution so step-0 behavior == the teacher.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def sample_ids(self, batch: int, num_tracks: int, device: torch.device,
                   generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(0, self.max_track_id, (batch, num_tracks), device=device, generator=generator)

    def forward(self,
                coords: torch.Tensor,
                visibility: torch.Tensor,
                latent_t: int,
                latent_h: int,
                latent_w: int,
                track_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            coords:     (B, T, N, 2) normalized to [0, 1] as (x, y).
            visibility: (B, T, N), 1 for visible/active, 0 for occluded/unspecified.
            latent_t/h/w: target latent dims (post VAE temporal/spatial compression).
            track_ids:  optional (B, N) integer IDs; sampled randomly if None.
        Returns:
            (B, track_channels, latent_t, latent_h, latent_w)
        """
        B, T, N, _ = coords.shape
        device = coords.device
        if track_ids is None:
            track_ids = self.sample_ids(B, N, device)

        phi = sinusoidal_embedding(track_ids, self.id_dim)  # (B, N, id_dim) float32

        coords_f = coords.float()
        x = (coords_f[..., 0].clamp(0.0, 1.0) * (latent_w - 1)).round().long()  # (B, T, N)
        y = (coords_f[..., 1].clamp(0.0, 1.0) * (latent_h - 1)).round().long()
        vis = visibility.float()  # (B, T, N)

        # Scatter v[t,n] * phi_n into the latent-grid cell (snap-to-block).
        contrib = vis.unsqueeze(-1) * phi.unsqueeze(1)  # (B, T, N, id_dim)
        cell = (y * latent_w + x)  # (B, T, N) flat cell index in [0, H*W)
        cm = torch.zeros((B, T, latent_h * latent_w, self.id_dim), device=device, dtype=torch.float32)
        cm.scatter_add_(2, cell.unsqueeze(-1).expand(-1, -1, -1, self.id_dim), contrib)
        cm = cm.view(B, T, latent_h, latent_w, self.id_dim).permute(0, 4, 1, 2, 3).contiguous()
        # (B, id_dim, T, H, W)

        cm = cm.to(self.temporal_conv.weight.dtype)
        # Causal VAE time mapping: T_lat = (T - 1) // t_comp + 1 -> left-pad time by (t_comp - 1).
        cm = F.pad(cm, (0, 0, 0, 0, self.vae_temporal_compression - 1, 0))
        cm = self.temporal_conv(cm)  # (B, track_channels, ~T_lat, H, W)
        if cm.shape[2] != latent_t:
            cm = F.interpolate(cm, size=(latent_t, latent_h, latent_w), mode="nearest")
        cm = self.proj(cm)
        return cm
