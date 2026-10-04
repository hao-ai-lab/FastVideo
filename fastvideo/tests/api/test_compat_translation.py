# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``fastvideo.api.compat`` keyword and request helpers, and
for how resolution carries the typed CompileConfig and
PipelineSelection.vae_tiling surfaces promoted in PR 6.
"""
from __future__ import annotations

import pytest

from fastvideo.api.compat import (
    FROM_PRETRAINED_KWARGS,
    _FROM_PRETRAINED_FIELDS,
    from_pretrained_kwargs_to_config,
    request_to_sampling_param,
)
from fastvideo.api.inference_resolution import resolve_inference_config, torch_compile_kwargs
from fastvideo.api.parser import parse_config
from fastvideo.api.schema import CompileConfig, GenerationRequest, GeneratorConfig
from fastvideo.api.sampling_param import SamplingParam


class TestFromPretrainedKwargsTranslation:
    """The ``from_pretrained`` keyword ``torch_compile_kwargs={...}`` gets
    split across the four first-class :class:`CompileConfig` fields and
    anything unknown falls into ``extras``. Keywords outside the
    ``from_pretrained`` set are rejected."""

    def test_empty_kwargs_produces_empty_extras(self) -> None:
        config = from_pretrained_kwargs_to_config(
            "/models/ltx2",
            {"torch_compile_kwargs": {}},
        )
        compile_config = config.engine.compile
        assert compile_config.extras == {}
        assert compile_config.backend is None

    def test_other_keyword_points_to_from_config(self) -> None:
        with pytest.raises(TypeError, match=r"does not accept ltx2_vae_tiling\. It accepts only these keywords: "
                           r"(.|\n)*VideoGenerator\.from_config"):
            from_pretrained_kwargs_to_config("/models/ltx2", {"ltx2_vae_tiling": True})

    def test_keyword_fields_are_the_from_pretrained_keywords(self) -> None:
        """Every keyword with a schema field sets one field; ``torch_compile_kwargs`` and ``nvfp4_fa4`` have none."""
        assert set(_FROM_PRETRAINED_FIELDS) == FROM_PRETRAINED_KWARGS - {"torch_compile_kwargs", "nvfp4_fa4"}
        assert len(set(_FROM_PRETRAINED_FIELDS.values())) == len(_FROM_PRETRAINED_FIELDS)
        config = from_pretrained_kwargs_to_config("/models/wan", {"dit_cpu_offload": False, "sp_size": 2})
        assert config.engine.offload.dit is False and config.engine.parallelism.sp_size == 2


class TestCompileConfigRoundTrip:
    """typed CompileConfig -> ``torch_compile_kwargs(resolved_config)``
    reconstruction drops ``None`` typed fields and merges ``extras``."""

    def test_only_typed_fields_emitted(self, monkeypatch) -> None:
        _skip_model_definition(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig(enabled=True, backend="inductor", fullgraph=True)),
        )
        resolved = resolve_inference_config(config)
        assert resolved.engine.compile.enabled is True
        assert torch_compile_kwargs(resolved) == {
            "backend": "inductor",
            "fullgraph": True,
        }

    def test_extras_merged_into_torch_compile_kwargs(self, monkeypatch) -> None:
        _skip_model_definition(monkeypatch)
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
        resolved = resolve_inference_config(config)
        assert torch_compile_kwargs(resolved) == {
            "mode": "reduce-overhead",
            "options": {
                "triton.cudagraphs": False
            },
        }

    def test_none_fields_suppressed(self, monkeypatch) -> None:
        _skip_model_definition(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        resolved = resolve_inference_config(config)
        assert torch_compile_kwargs(resolved) == {}


class TestVaeTilingResolution:
    """``generator.pipeline.vae_tiling`` reaches the resolved config at
    its typed path, and stays unset when neither the input nor a model
    default sets it."""

    def test_explicit_value_is_kept(self, monkeypatch) -> None:
        _skip_model_definition(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        config.pipeline.vae_tiling = False
        resolved = resolve_inference_config(config)
        assert resolved.pipeline.vae_tiling is False

    def test_unset_stays_unset(self, monkeypatch) -> None:
        _skip_model_definition(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        resolved = resolve_inference_config(config)
        assert resolved.pipeline.vae_tiling is None


class TestTextEncoderCompileResolution:
    """``generator.engine.compile.text_encoder_enabled`` keeps an explicit
    value, and resolution gives an unset value its runtime default."""

    def test_explicit_value_is_kept(self, monkeypatch) -> None:
        _skip_model_definition(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig(text_encoder_enabled=True)),
        )
        resolved = resolve_inference_config(config)
        assert resolved.engine.compile.text_encoder_enabled is True

    def test_unset_takes_the_runtime_default(self, monkeypatch) -> None:
        _skip_model_definition(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        resolved = resolve_inference_config(config)
        assert resolved.engine.compile.text_encoder_enabled is False


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


def _skip_model_definition(monkeypatch):
    """Skip the model definition (registry lookup, model defaults, and
    PipelineConfig materialization), so resolution tests need no resolvable
    model path."""
    from fastvideo.api import inference_resolution

    monkeypatch.setattr(inference_resolution, "build_model_pipeline_config", lambda config: None)
    monkeypatch.setattr(inference_resolution, "pipeline_config_defaults_step", lambda config, defaults=None: lambda view: {})
    monkeypatch.setattr(inference_resolution, "materialize_pipeline_config", lambda resolved, pipeline_config: None)
