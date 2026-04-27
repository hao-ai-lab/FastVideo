# SPDX-License-Identifier: Apache-2.0
"""Typed quantization flow contract tests.

Locks in the path from typed
``GeneratorConfig.engine.quantization.transformer_quant: "NVFP4"``
through the compat layer to a concrete ``NVFP4Config`` instance pinned
on ``pipeline_config.dit_config.quant_config``.

The model loader detects FP4 by ``isinstance(quant_method,
NVFP4QuantizeMethod)`` rather than by a flag, so the typed surface
must reliably produce that class on the DiT config — otherwise the
loader silently runs full bf16.
"""
from __future__ import annotations

import pytest

from fastvideo.api.compat import generator_config_to_fastvideo_args
from fastvideo.api.schema import (
    EngineConfig,
    GeneratorConfig,
    QuantizationConfig,
)
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config


@pytest.fixture
def captured_kwargs(monkeypatch):
    """Replace ``FastVideoArgs.from_kwargs`` with a capturer so the
    test doesn't try to download model_index.json.
    """
    from fastvideo import fastvideo_args as fva

    captured: dict[str, object] = {}

    def _capture(**kw):
        captured.update(kw)

        class _Stub:
            kwargs = kw

        return _Stub()

    monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _capture)
    return captured


def test_typed_transformer_quant_resolves_to_nvfp4_instance(
        captured_kwargs) -> None:
    cfg = GeneratorConfig(
        model_path="FastVideo/LTX2-Distilled-Diffusers",
        engine=EngineConfig(
            quantization=QuantizationConfig(transformer_quant="NVFP4"), ),
    )
    generator_config_to_fastvideo_args(cfg)
    assert "transformer_quant" in captured_kwargs, (
        "compat layer must forward typed transformer_quant onto "
        "FastVideoArgs.from_kwargs")
    assert isinstance(captured_kwargs["transformer_quant"], NVFP4Config), (
        f"Expected NVFP4Config instance, got "
        f"{type(captured_kwargs['transformer_quant']).__name__}")


def test_no_typed_quant_omits_transformer_quant_kwarg(captured_kwargs) -> None:
    """Default GeneratorConfig has ``quantization=None`` — the carrier
    must not be set, so the existing legacy path
    (``pipeline_config.dit_config.quant_config = NVFP4Config()``)
    keeps working as before.
    """
    cfg = GeneratorConfig(model_path="FastVideo/LTX2-Distilled-Diffusers")
    generator_config_to_fastvideo_args(cfg)
    assert "transformer_quant" not in captured_kwargs


def test_apply_transformer_quant_pins_to_dit_config(monkeypatch) -> None:
    """``FastVideoArgs.__post_init__._apply_transformer_quant`` must
    copy the ``transformer_quant`` instance onto
    ``pipeline_config.dit_config.quant_config`` so the DiT loader sees
    it during construction.
    """
    from fastvideo.fastvideo_args import FastVideoArgs

    args = FastVideoArgs(model_path="FastVideo/LTX2-Distilled-Diffusers")
    # ``transformer_quant`` defaults to None so __post_init__ leaves it.
    assert args.transformer_quant is None
    assert args.pipeline_config.dit_config.quant_config is None

    nvfp4 = NVFP4Config()
    args.transformer_quant = nvfp4
    args._apply_transformer_quant()
    assert args.pipeline_config.dit_config.quant_config is nvfp4


def test_apply_transformer_quant_does_not_overwrite_explicit_dit_config(
) -> None:
    """When the caller has explicitly set
    ``pipeline_config.dit_config.quant_config`` already, the typed
    carrier defers — the explicit setter wins.
    """
    from fastvideo.fastvideo_args import FastVideoArgs

    explicit = NVFP4Config(layer_profile="base")
    args = FastVideoArgs(model_path="FastVideo/LTX2-Distilled-Diffusers")
    args.pipeline_config.dit_config.quant_config = explicit
    args.transformer_quant = NVFP4Config(layer_profile="refine")
    args._apply_transformer_quant()
    assert args.pipeline_config.dit_config.quant_config is explicit
