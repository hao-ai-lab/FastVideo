# SPDX-License-Identifier: Apache-2.0
"""Typed quantization flow contract tests.

Locks in the path from typed
``GeneratorConfig.engine.quantization.transformer_quant: "NVFP4"``
through resolution to a concrete ``NVFP4Config`` instance pinned
on ``pipeline_config.dit_config.quant_config``.

The model loader detects FP4 by ``isinstance(quant_method,
NVFP4QuantizeMethod)`` rather than by a flag, so the typed surface
must reliably produce that class on the DiT config — otherwise the
loader silently runs full bf16.
"""
from __future__ import annotations

from types import SimpleNamespace

from fastvideo.api.inference_resolution import _apply_transformer_quant, resolve_inference_config
from fastvideo.api.schema import (
    EngineConfig,
    GeneratorConfig,
    QuantizationConfig,
)
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
from fastvideo.tests.api.config_snapshot import isolated_environment

LTX2 = "FastVideo/LTX2-Distilled-Diffusers"


def _resolve(config: GeneratorConfig):
    with isolated_environment():
        return resolve_inference_config(config)


def test_typed_transformer_quant_resolves_to_nvfp4_instance() -> None:
    resolved = _resolve(
        GeneratorConfig(
            model_path=LTX2,
            engine=EngineConfig(quantization=QuantizationConfig(transformer_quant="NVFP4"), ),
        ))
    quant_config = resolved.pipeline_config.dit_config.quant_config
    assert isinstance(quant_config, NVFP4Config), (f"Expected NVFP4Config instance, got "
                                                   f"{type(quant_config).__name__}")


def test_no_typed_quant_leaves_the_dit_quant_config_unset() -> None:
    """Default GeneratorConfig has ``quantization=None``, so resolution
    pins nothing and ``pipeline_config.dit_config.quant_config`` keeps
    the model default.
    """
    resolved = _resolve(GeneratorConfig(model_path=LTX2))
    assert resolved.pipeline_config.dit_config.quant_config is None


def test_apply_transformer_quant_pins_to_dit_config() -> None:
    """Materialization must pin the config that ``transformer_quant`` names
    on ``pipeline_config.dit_config.quant_config`` so the DiT loader sees
    it during construction.
    """
    pipeline_config = SimpleNamespace(dit_config=SimpleNamespace(quant_config=None))

    _apply_transformer_quant(pipeline_config, "NVFP4")
    assert isinstance(pipeline_config.dit_config.quant_config, NVFP4Config)


def test_apply_transformer_quant_does_not_overwrite_explicit_dit_config() -> None:
    """When ``pipeline_config.dit_config.quant_config`` is already set,
    the typed carrier defers — the explicit setter wins.
    """
    explicit = NVFP4Config(layer_profile="base")
    pipeline_config = SimpleNamespace(dit_config=SimpleNamespace(quant_config=explicit))
    _apply_transformer_quant(pipeline_config, "NVFP4")
    assert pipeline_config.dit_config.quant_config is explicit
