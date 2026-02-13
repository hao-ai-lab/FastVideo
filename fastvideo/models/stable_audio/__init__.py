# SPDX-License-Identifier: Apache-2.0
"""Stable Audio: pretransform (VAE), conditioner, sampling, DiT.

All components are in-repo; no stable-audio-tools clone required.
- Sampling: k-diffusion only. Conditioner: t5 + number (conditioners_inline).
- Pretransform: Oobleck VAE (sat_factory). DiT: DiffusionTransformer (sat_dit).
"""
from .pretransform import StableAudioPretransform
from .conditioner import StableAudioConditioner

__all__ = [
    "StableAudioPretransform",
    "StableAudioConditioner",
]
