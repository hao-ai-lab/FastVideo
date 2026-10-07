# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 audio VAE configuration.

Mirrors the checkpoint's audio_vae/config.json:
{
  "_class_name": "MMAudioVAE", "mode": "44k", "sample_rate": 44100,
  "downsample_factor": 1024, "scaling_factor": 0.5302,
  "vocoder_config": {...BigVGAN-v2 hyperparams, ignored here -- redundant
  backward-compat copy; the pipeline loads a separate "vocoder" component
  instead, see fastvideo/models/audio/kandinsky6_audio_vae.py}
}
"""

from dataclasses import dataclass, field

from fastvideo.configs.models.base import ArchConfig, ModelConfig


@dataclass
class Kandinsky6AudioVAEArchConfig(ArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["Kandinsky6AudioVAE"])
    mode: str = "44k"
    need_vae_decoder: bool = True
    need_vae_encoder: bool = False
    scaling_factor: float = 1.0


@dataclass
class Kandinsky6AudioVAEConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=Kandinsky6AudioVAEArchConfig)
