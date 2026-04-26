# SPDX-License-Identifier: Apache-2.0
"""Smoke / preflight tests for the Stable Audio Open 1.0 T2A pipeline."""
from __future__ import annotations

import os

import pytest


def test_stable_audio_typed_surface_preflight() -> None:
    """No-GPU preflight: imports + registry + preset are wired.
    Catches the kind of regressions that otherwise only surface on a
    GPU host (preset dropped from ALL_PRESETS, registry mis-wired,
    EntryClass renamed, etc.).
    """
    import fastvideo.registry  # noqa: F401
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.pipelines.stable_audio import StableAudioT2AConfig
    from fastvideo.pipelines.basic.stable_audio.stable_audio_pipeline import (
        EntryClass,
        StableAudioPipeline,
    )
    from fastvideo.pipelines.basic.stable_audio.stages import (  # noqa: F401
        StableAudioConditioningStage,
        StableAudioDecodingStage,
        StableAudioDenoisingStage,
        StableAudioLatentPreparationStage,
    )

    assert EntryClass is StableAudioPipeline

    names = {p.name for p in get_presets_for_family("stable_audio")}
    assert names == {"stable_audio_open_1_0_base"}
    preset = get_preset("stable_audio_open_1_0_base", "stable_audio")
    assert preset.defaults["num_inference_steps"] == 100
    assert preset.defaults["guidance_scale"] == 7.0

    pc = StableAudioT2AConfig()
    assert pc.sampling_rate == 44100
    assert pc.audio_channels == 2
    assert pc.guidance_scale == 7.0
    assert pc.num_inference_steps == 100
    # First-class OobleckVAE is wired into the VAE slot.
    from fastvideo.configs.models.vaes import OobleckVAEConfig
    assert isinstance(pc.vae_config, OobleckVAEConfig)
    assert pc.vae_config.pretrained_path == "stabilityai/stable-audio-open-1.0"
