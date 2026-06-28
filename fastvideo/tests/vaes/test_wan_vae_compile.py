# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Wan VAE torch.compile wiring.

Regression coverage for the fix that makes ``enable_torch_compile_vae`` actually
compile ``AutoencoderKLWan``. Without ``_compile_conditions`` the pipeline takes
the full-module-replace fallback in ``_maybe_compile_pipeline_module`` and the
compiled VAE is never used by the already-constructed ``DecodingStage`` (no-op).

These tests are intentionally lightweight (no GPU, no weights): they only check
that the compile-condition predicate is registered on the class and matches the
encoder/decoder submodules by name and type.
"""
import torch.nn as nn

from fastvideo.models.vaes.wanvae import (AutoencoderKLWan, WanDecoder3d,
                                          WanEncoder3d, _is_wan_vae_codec)


def test_wan_vae_registers_compile_conditions():
    # The class must declare _compile_conditions so _compile_with_conditions
    # compiles the codec in place instead of falling back to a no-op replace.
    conditions = getattr(AutoencoderKLWan, "_compile_conditions", None)
    assert conditions, "AutoencoderKLWan must declare _compile_conditions"
    assert _is_wan_vae_codec in conditions


def test_is_wan_vae_codec_matches_encoder_decoder():
    # object.__new__ gives a correctly-typed instance without running __init__,
    # so the isinstance check can be exercised without a config or GPU.
    decoder = object.__new__(WanDecoder3d)
    encoder = object.__new__(WanEncoder3d)

    assert _is_wan_vae_codec("decoder", decoder) is True
    assert _is_wan_vae_codec("encoder", encoder) is True


def test_is_wan_vae_codec_rejects_other_submodules():
    decoder = object.__new__(WanDecoder3d)
    encoder = object.__new__(WanEncoder3d)

    # Right name but wrong type (e.g. a small conv/linear at the same name).
    assert _is_wan_vae_codec("decoder", nn.Linear(1, 1)) is False
    # Right type but not the top-level codec name.
    assert _is_wan_vae_codec("decoder.something", decoder) is False
    assert _is_wan_vae_codec("post_quant_conv", decoder) is False
    # Mismatched name and type (each name must match its own codec type).
    assert _is_wan_vae_codec("encoder", decoder) is False
    assert _is_wan_vae_codec("decoder", encoder) is False
