# SPDX-License-Identifier: Apache-2.0
"""Tests for ``fastvideo.api.compat`` translation helpers covering the
typed CompileConfig + PipelineSelection.vae_tiling surfaces promoted in
PR 6.
"""
from __future__ import annotations

from fastvideo.api.compat import (
    generator_config_to_fastvideo_args,
    legacy_from_pretrained_to_config,
)
from fastvideo.api.schema import CompileConfig, GeneratorConfig


class TestLegacyTorchCompileKwargsTranslation:
    """Legacy ``torch_compile_kwargs={...}`` gets split across the four
    first-class :class:`CompileConfig` fields and anything unknown falls
    into ``extras``."""

    def test_all_typed_keys_promoted(self) -> None:
        config = legacy_from_pretrained_to_config(
            "/models/ltx2",
            {
                "enable_torch_compile": True,
                "torch_compile_kwargs": {
                    "backend": "inductor",
                    "fullgraph": True,
                    "mode": "max-autotune-no-cudagraphs",
                    "dynamic": False,
                },
            },
        )
        compile_config = config.engine.compile
        assert compile_config.enabled is True
        assert compile_config.backend == "inductor"
        assert compile_config.fullgraph is True
        assert compile_config.mode == "max-autotune-no-cudagraphs"
        assert compile_config.dynamic is False
        assert compile_config.extras == {}

    def test_unknown_keys_land_in_extras(self) -> None:
        config = legacy_from_pretrained_to_config(
            "/models/ltx2",
            {
                "enable_torch_compile": True,
                "torch_compile_kwargs": {
                    "backend": "inductor",
                    "options": {"triton.cudagraphs": False},
                    "disable": False,
                },
            },
        )
        compile_config = config.engine.compile
        assert compile_config.backend == "inductor"
        assert compile_config.extras == {
            "options": {"triton.cudagraphs": False},
            "disable": False,
        }

    def test_empty_kwargs_produces_empty_extras(self) -> None:
        config = legacy_from_pretrained_to_config(
            "/models/ltx2",
            {"torch_compile_kwargs": {}},
        )
        compile_config = config.engine.compile
        assert compile_config.extras == {}
        assert compile_config.backend is None


class TestCompileConfigRoundTrip:
    """typed CompileConfig -> FastVideoArgs.torch_compile_kwargs
    reconstruction drops ``None`` typed fields and merges ``extras``."""

    def test_only_typed_fields_emitted(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(
                CompileConfig(enabled=True, backend="inductor", fullgraph=True)),
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
                    extras={"options": {"triton.cudagraphs": False}},
                )),
        )
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["torch_compile_kwargs"] == {
            "mode": "reduce-overhead",
            "options": {"triton.cudagraphs": False},
        }

    def test_none_fields_suppressed(self, monkeypatch) -> None:
        _stub_fastvideo_args_from_kwargs(monkeypatch)
        config = GeneratorConfig(
            model_path="/models/ltx2",
            engine=_engine_with_compile(CompileConfig()),
        )
        args = generator_config_to_fastvideo_args(config)
        assert args.kwargs["torch_compile_kwargs"] == {}


class TestLegacyLtx2VaeTilingTranslation:
    """``ltx2_vae_tiling`` flat kwarg promotes to
    ``generator.pipeline.vae_tiling``; reverse direction emits the
    legacy name back to FastVideoArgs."""

    def test_forward_routes_to_pipeline_vae_tiling(self) -> None:
        config = legacy_from_pretrained_to_config(
            "/models/ltx2",
            {"ltx2_vae_tiling": False},
        )
        assert config.pipeline.vae_tiling is False

    def test_true_round_trips(self) -> None:
        config = legacy_from_pretrained_to_config(
            "/models/ltx2",
            {"ltx2_vae_tiling": True},
        )
        assert config.pipeline.vae_tiling is True

    def test_unset_stays_none(self) -> None:
        config = legacy_from_pretrained_to_config("/models/ltx2", {})
        assert config.pipeline.vae_tiling is None

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
    """Swap ``FastVideoArgs.from_kwargs`` for a capture-only stub so
    translation tests don't need to construct a valid FastVideoArgs."""
    from fastvideo import fastvideo_args as fva

    class _Captured:

        def __init__(self, **kw):
            self.kwargs = kw

    monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _Captured)
