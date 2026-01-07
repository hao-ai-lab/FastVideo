# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 video VAE wrappers using the official ltx-core implementation.
"""

from pathlib import Path
import sys
from typing import Any

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
            VideoDecoder,
            VideoDecoderConfigurator,
            VideoEncoder,
            VideoEncoderConfigurator,
        )
    except ImportError as exc:
        raise ImportError(
            "LTX-2 sources are required. Ensure FastVideo/LTX-2 is present."
        ) from exc

    return VideoDecoder, VideoDecoderConfigurator, VideoEncoder, VideoEncoderConfigurator


class LTX2VideoEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        VideoDecoder, VideoDecoderConfigurator, VideoEncoder, VideoEncoderConfigurator = _require_ltx2()
        self.model: VideoEncoder = VideoEncoderConfigurator.from_config(config)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.model(sample)


class LTX2VideoDecoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        VideoDecoder, VideoDecoderConfigurator, VideoEncoder, VideoEncoderConfigurator = _require_ltx2()
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
        VideoDecoder, VideoDecoderConfigurator, VideoEncoder, VideoEncoderConfigurator = _require_ltx2()

        self.encoder = VideoEncoderConfigurator.from_config(config)
        self.decoder = VideoDecoderConfigurator.from_config(config)

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
        return self.decoder(z, timestep=timestep, generator=generator)

    def enable_tiling(self) -> None:
        # LTX-2 VAE does not implement FastVideo tiling; no-op to satisfy pipeline calls.
        return None
