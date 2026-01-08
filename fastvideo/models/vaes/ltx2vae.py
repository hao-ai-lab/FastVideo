# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 video VAE wrappers using the official ltx-core implementation.
"""

from pathlib import Path
import sys
from typing import Any, Iterable

import torch
import torch.nn as nn

from fastvideo.models.vaes.common import DiagonalGaussianDistribution


def _require_ltx2():
    repo_root = Path(__file__).resolve().parents[3]
    local_core = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
    if local_core.exists() and str(local_core) not in sys.path:
        sys.path.insert(0, str(local_core))

    try:
        from ltx_core.model.video_vae import (  # type: ignore
            SpatialTilingConfig,
            TemporalTilingConfig,
            TilingConfig,
            VideoDecoder,
            VideoDecoderConfigurator,
            VideoEncoder,
            VideoEncoderConfigurator,
        )
    except ImportError as exc:
        raise ImportError(
            "LTX-2 sources are required. Ensure FastVideo/LTX-2 is present."
        ) from exc

    return (
        SpatialTilingConfig,
        TemporalTilingConfig,
        TilingConfig,
        VideoDecoder,
        VideoDecoderConfigurator,
        VideoEncoder,
        VideoEncoderConfigurator,
    )


def _concat_tiled_chunks(chunks: Iterable[torch.Tensor]) -> torch.Tensor:
    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("No chunks produced by tiled decode.")
    if len(chunk_list) == 1:
        return chunk_list[0]
    return torch.cat(chunk_list, dim=2)


class LTX2VideoEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        (SpatialTilingConfig, TemporalTilingConfig, TilingConfig, VideoDecoder,
         VideoDecoderConfigurator, VideoEncoder,
         VideoEncoderConfigurator) = _require_ltx2()
        self.model: VideoEncoder = VideoEncoderConfigurator.from_config(config)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.model(sample)


class LTX2VideoDecoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        (SpatialTilingConfig, TemporalTilingConfig, TilingConfig, VideoDecoder,
         VideoDecoderConfigurator, VideoEncoder,
         VideoEncoderConfigurator) = _require_ltx2()
        self.model: VideoDecoder = VideoDecoderConfigurator.from_config(config)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.model(sample, timestep=timestep, generator=generator)


class LTX2CausalVideoAutoencoder(nn.Module):
    """
    LTX-2 VAE wrapper that exposes FastVideo's VAE encode/decode interface.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        (self._SpatialTilingConfig, self._TemporalTilingConfig,
         self._TilingConfig, VideoDecoder, VideoDecoderConfigurator,
         VideoEncoder, VideoEncoderConfigurator) = _require_ltx2()

        self.encoder = VideoEncoderConfigurator.from_config(config)
        self.decoder = VideoDecoderConfigurator.from_config(config)
        self._tiling_config: Any | None = None
        self._use_tiling: bool = False

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        means = self.encoder(x)
        zeros = torch.zeros_like(means)
        return DiagonalGaussianDistribution(torch.cat([means, zeros], dim=1), deterministic=True)

    def decode(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self._use_tiling:
            if self._tiling_config is None:
                self._tiling_config = self._TilingConfig.default()
            chunks = self.decoder.tiled_decode(
                z,
                tiling_config=self._tiling_config,
                timestep=timestep,
                generator=generator,
            )
            return _concat_tiled_chunks(chunks)
        return self.decoder(z, timestep=timestep, generator=generator)

    def enable_tiling(self) -> None:
        self._use_tiling = True
        if self._tiling_config is None:
            self._tiling_config = self._TilingConfig.default()

    def set_tiling_config(
        self,
        spatial_tile_size_in_pixels: int = 512,
        spatial_tile_overlap_in_pixels: int = 64,
        temporal_tile_size_in_frames: int = 64,
        temporal_tile_overlap_in_frames: int = 24,
    ) -> None:
        self._tiling_config = self._TilingConfig(
            spatial_config=self._SpatialTilingConfig(
                tile_size_in_pixels=spatial_tile_size_in_pixels,
                tile_overlap_in_pixels=spatial_tile_overlap_in_pixels,
            ),
            temporal_config=self._TemporalTilingConfig(
                tile_size_in_frames=temporal_tile_size_in_frames,
                tile_overlap_in_frames=temporal_tile_overlap_in_frames,
            ),
        )
