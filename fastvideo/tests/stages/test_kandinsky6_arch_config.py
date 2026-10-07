# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Kandinsky6ArchConfig's __post_init__ validation.

Pure dataclass/logic tests -- no GPU, no model weights.
"""
from __future__ import annotations

import pytest

from fastvideo.configs.models.dits.kandinsky6 import Kandinsky6ArchConfig


def test_defaults_match_diffusers_reference_not_kandinsky5():
    # Kandinsky5ArchConfig's defaults are much smaller (in_visual_dim=4,
    # model_dim=2048, num_visual_blocks=32, axes_dims=(16,24,24)) -- this
    # pins Kandinsky6ArchConfig to the actual Kandinsky6Transformer3DModel
    # defaults so a future edit can't silently drift back toward Kandinsky5's.
    cfg = Kandinsky6ArchConfig()
    assert cfg.in_visual_dim == 16
    assert cfg.out_visual_dim == 16
    assert cfg.in_text_dim == 3584
    assert cfg.in_text_dim2 == 768
    assert cfg.time_dim == 1024
    assert cfg.model_dim == 4096
    assert cfg.ff_dim == 16384
    assert cfg.num_text_blocks == 4
    assert cfg.num_visual_blocks == 60
    assert cfg.axes_dims == (32, 48, 48)
    assert cfg.in_audio_dim == 20
    # Both official checkpoints (Pro-sft and Pro-distill) declare this fixed value in
    # transformer/config.json; it is not derived from height/width.
    assert cfg.scale_factor == (1.0, 2.0, 2.0)


def test_audio_dims_default_to_matching_video_dims():
    cfg = Kandinsky6ArchConfig(model_dim=48, time_dim=32, ff_dim=128, axes_dims=(8, 8, 8))
    assert cfg.model_dim_a == 48
    assert cfg.time_dim_a == 32
    assert cfg.ff_dim_a == 128
    assert cfg.axes_dims_a == (8, 8, 8)


def test_audio_dims_can_be_overridden_independently():
    cfg = Kandinsky6ArchConfig(
        model_dim=48,
        time_dim=32,
        ff_dim=128,
        axes_dims=(8, 8, 8),
        model_dim_a=48,
        time_dim_a=32,  # must match time_dim unless fix_modulation=True
        ff_dim_a=96,
        axes_dims_a=(4, 4, 4),
    )
    assert cfg.model_dim_a == 48
    assert cfg.ff_dim_a == 96
    assert cfg.axes_dims_a == (4, 4, 4)


def test_model_dim_a_not_divisible_by_head_dim_a_raises():
    with pytest.raises(ValueError, match="model_dim_a"):
        Kandinsky6ArchConfig(
            model_dim=48,
            time_dim=32,
            axes_dims=(8, 8, 8),
            model_dim_a=50,  # 50 % sum((4,4,4))=12 != 0
            time_dim_a=32,
            axes_dims_a=(4, 4, 4),
        )


def test_mismatched_time_dim_a_without_fix_modulation_raises():
    # Kandinsky6FusedTransformerDecoderBlock's cross-modal modulation is
    # driven by the *other* modality's time embedding by default
    # (fix_modulation=False) -- va_modulation is built from time_dim but
    # invoked with the audio time embedding, so time_dim_a must equal
    # time_dim or the fused block's modulation matmul shapes mismatch.
    with pytest.raises(ValueError, match="time_dim_a"):
        Kandinsky6ArchConfig(model_dim=48, time_dim=32, axes_dims=(8, 8, 8), time_dim_a=16)


def test_mismatched_time_dim_a_allowed_with_fix_modulation():
    cfg = Kandinsky6ArchConfig(model_dim=48, time_dim=32, axes_dims=(8, 8, 8), time_dim_a=16,
                               fix_modulation=True)
    assert cfg.time_dim_a == 16


def test_is_multimodal_false_is_rejected():
    # The DiT is always the joint video+audio model; a config asking for anything else fails loudly.
    with pytest.raises(ValueError, match="is_multimodal=True"):
        Kandinsky6ArchConfig(model_dim=48, time_dim=32, axes_dims=(8, 8, 8), is_multimodal=False)


def test_derived_arch_fields_match_dit_arch_config_contract():
    cfg = Kandinsky6ArchConfig(model_dim=48, time_dim=32, axes_dims=(8, 8, 8), in_visual_dim=6, out_visual_dim=6)
    assert cfg.hidden_size == 48
    assert cfg.num_attention_heads == 48 // 24
    assert cfg.in_channels == 6
    assert cfg.out_channels == 6
    assert cfg.num_channels_latents == 6
