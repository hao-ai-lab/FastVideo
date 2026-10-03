# SPDX-License-Identifier: Apache-2.0
"""Tests for ``fastvideo.api.compat`` translation helpers covering the
typed CompileConfig + PipelineSelection.vae_tiling surfaces promoted in
PR 6.
"""
from __future__ import annotations

import pytest

from fastvideo.api.compat import (
    from_pretrained_kwargs_to_config,
    generator_config_to_fastvideo_args,
    request_to_sampling_param,
)
from fastvideo.api.parser import parse_config
from fastvideo.api.schema import CompileConfig, GenerationRequest, GeneratorConfig
from fastvideo.api.sampling_param import SamplingParam


class TestFromPretrainedKwargsTranslation:
    """The ``from_pretrained`` keyword ``torch_compile_kwargs={...}`` gets
    split across the four first-class :class:`CompileConfig` fields and
    anything unknown falls into ``extras``. Keywords outside the
    ``from_pretrained`` set are rejected with their typed path."""

    def test_empty_kwargs_produces_empty_extras(self) -> None:
        config = from_pretrained_kwargs_to_config(
            "/models/ltx2",
            {"torch_compile_kwargs": {}},
        )
        compile_config = config.engine.compile
        assert compile_config.extras == {}
        assert compile_config.backend is None

    def test_other_keyword_names_its_typed_path(self) -> None:
        with pytest.raises(TypeError, match="ltx2_vae_tiling -> pipeline.vae_tiling"):
            from_pretrained_kwargs_to_config("/models/ltx2", {"ltx2_vae_tiling": True})


class TestCompileConfigRoundTrip:
    """typed CompileConfig -> FastVideoArgs.torch_compile_kwargs
    reconstruction drops ``None`` typed fields and merges ``extras``."""

    def test_only_typed_fields_emitted(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig(enabled=True, backend="inductor", fullgraph=True)),
        )
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["enable_torch_compile"] is True
        assert args.kwargs["torch_compile_kwargs"] == {
            "backend": "inductor",
            "fullgraph": True,
        }

    def test_extras_merged_into_torch_compile_kwargs(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(
                CompileConfig(
                    enabled=True,
                    mode="reduce-overhead",
                    extras={"options": {
                        "triton.cudagraphs": False
                    }},
                )),
        )
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["torch_compile_kwargs"] == {
            "mode": "reduce-overhead",
            "options": {
                "triton.cudagraphs": False
            },
        }

    def test_none_fields_suppressed(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["torch_compile_kwargs"] == {}


class TestLtx2VaeTilingFlattening:
    """``generator.pipeline.vae_tiling`` reaches FastVideoArgs as the flat
    keyword ``ltx2_vae_tiling``."""

    def test_reverse_emits_legacy_name(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        config.pipeline.vae_tiling = False
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["ltx2_vae_tiling"] is False

    def test_reverse_unset_skips_key(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        args = generator_config_to_fastvideo_args(config)
        assert "ltx2_vae_tiling" not in args.kwargs


class TestTextEncoderCompileFlattening:
    """``generator.engine.compile.text_encoder_enabled`` reaches the
    FastVideoArgs kwargs dict as ``enable_torch_compile_text_encoder`` so
    realtime-runtime consumers can read it before FastVideoArgs filters
    unknown fields."""

    def test_reverse_emits_legacy_name(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig(text_encoder_enabled=True)),
        )
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["enable_torch_compile_text_encoder"] is True

    def test_reverse_unset_emits_the_runtime_default(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["enable_torch_compile_text_encoder"] is False


def test_batch_cfg_typed_request_reaches_sampling_param(monkeypatch) -> None:
    """Preserve explicit batched CFG through the canonical request adapter."""
    monkeypatch.setattr(
        SamplingParam,
        "from_pretrained",
        classmethod(lambda cls, model_path: cls()),
    )
    request = parse_config(
        GenerationRequest,
        {
            "prompt": "fox",
            "sampling": {
                "batch_cfg": True
            }
        },
    )
    sampling_param = request_to_sampling_param(request, model_path="test-model")
    assert sampling_param.batch_cfg is True


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _engine_with_compile(compile_config):
    """Build an ``EngineConfig`` that carries the supplied compile block."""
    from fastvideo.api.schema import EngineConfig
    engine = EngineConfig()
    engine.compile = compile_config
    return engine


def _stub_fastvideo_args_from_kwargs(monkeypatch):
    """Swap ``FastVideoArgs.from_kwargs`` for a capture-only stub, and skip the
    model definition (registry lookup, model defaults, and PipelineConfig
    materialization), so translation tests need neither a valid FastVideoArgs
    nor a resolvable model path."""
    from fastvideo import fastvideo_args as fva
    from fastvideo.api import inference_resolution

    monkeypatch.setattr(inference_resolution, "build_model_pipeline_config", lambda config: None)
    monkeypatch.setattr(inference_resolution, "pipeline_config_defaults_step", lambda config, defaults=None: lambda view: {})
    monkeypatch.setattr(inference_resolution, "materialize_pipeline_config", lambda resolved, pipeline_config: None)

    class _Captured:

        def __init__(self, **kw):
            self.kwargs = kw

    monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _Captured)
