# SPDX-License-Identifier: Apache-2.0
"""MMAudio audio VAE configuration."""

from dataclasses import dataclass, field

from fastvideo.configs.models.base import ArchConfig, ModelConfig


@dataclass
class MMAudioVAEArchConfig(ArchConfig):
    """MMAudio mel-spectrogram VAE architecture.

    ``mode`` is the single source of truth: ``MMAudioVAE`` derives the mel
    dimension, the latent dimension, and the hidden width from it, so they are
    not duplicated here.
    """

    architectures: list[str] = field(default_factory=lambda: ["MMAudioVAE"])
    mode: str = "44k"
    need_encoder: bool = False


@dataclass
class MMAudioVAEConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=MMAudioVAEArchConfig)
