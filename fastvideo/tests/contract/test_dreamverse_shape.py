# SPDX-License-Identifier: Apache-2.0
"""Contract test: Dreamverse-style inputs normalize through the public
typed API without needing any private-only compatibility promise.

The private Dreamverse UI server (``FastVideo-internal/ui/ltx2-streaming/
server/gpu_pool.py``) has historically called
``VideoGenerator.from_pretrained(**load_kwargs)`` with a flat kwarg bag
containing LTX-2-specific names (``ltx2_refine_enabled``,
``ltx2_refine_upsampler_path``, etc.). PR 6 gave every one of those
kwargs a typed home under ``GeneratorConfig``.

This test makes sure:

1. The public typed API can represent everything Dreamverse currently
   passes at init time (``legacy_from_pretrained_to_config``).
2. The request-path Dreamverse uses (``generator.generate_video(**kwargs)``
   with per-segment flags) round-trips through the typed ``GenerationRequest``
   without reintroducing private-only fields at the public boundary.
3. Private-only Dreamverse fields that don't belong on the public
   surface either go to ``pipeline.experimental`` / ``request.extensions``
   (the documented escape hatch) or raise explicitly, rather than
   silently becoming part of the public compatibility promise.

Regression guard for the scoping rule in ``apirefactor.md`` §"Schema
Parity Requirement".
"""
from __future__ import annotations

import pytest

from fastvideo.api import (
    ComponentConfig,
    CompileConfig,
    GeneratorConfig,
    GenerationRequest,
)
from fastvideo.api.compat import (
    legacy_from_pretrained_to_config,
    legacy_generate_call_to_request,
    normalize_generation_request,
)


def _dreamverse_load_kwargs() -> dict:
    """The exact shape internal ``gpu_pool.py`` passes to
    ``VideoGenerator.from_pretrained(**load_kwargs)`` today."""
    return {
        "config_model_path": "/models/ltx2-config",
        "ltx2_refine_enabled": True,
        "ltx2_refine_upsampler_path": "/models/ltx2-refine",
        "ltx2_refine_lora_path": "/models/ltx2-refine-lora",
        "ltx2_refine_num_inference_steps": 2,
        "ltx2_refine_guidance_scale": 1.0,
        "ltx2_refine_add_noise": True,
        "ltx2_vae_tiling": True,
        "torch_compile_kwargs": {
            "backend": "inductor",
            "mode": "reduce-overhead",
            "fullgraph": True,
        },
        "dit_cpu_offload": False,
        "vae_cpu_offload": False,
        "text_encoder_cpu_offload": False,
        "pin_cpu_memory": True,
        "use_fsdp_inference": False,
        "enable_torch_compile": True,
    }


class TestDreamverseLoadKwargsShape:
    """Every current Dreamverse init-time kwarg must land on a typed
    field, not in the ``experimental`` escape hatch."""

    def test_all_kwargs_land_on_typed_fields(self):
        config = legacy_from_pretrained_to_config(
            "/models/ltx2", _dreamverse_load_kwargs())
        assert isinstance(config, GeneratorConfig)
        # None of the kwargs should have been routed to experimental.
        assert config.pipeline.experimental == {}, (
            "Dreamverse kwargs leaked into pipeline.experimental: "
            f"{config.pipeline.experimental}")

    def test_refine_enabled_reaches_preset_overrides(self):
        config = legacy_from_pretrained_to_config(
            "/models/ltx2", _dreamverse_load_kwargs())
        refine = config.pipeline.preset_overrides.get("refine") or {}
        assert refine.get("enabled") is True
        assert refine.get("add_noise") is True
        assert refine.get("num_inference_steps") == 2
        assert refine.get("guidance_scale") == 1.0

    def test_refine_assets_reach_component_config(self):
        config = legacy_from_pretrained_to_config(
            "/models/ltx2", _dreamverse_load_kwargs())
        assert isinstance(config.pipeline.components, ComponentConfig)
        assert (config.pipeline.components.upsampler_weights ==
                "/models/ltx2-refine")
        assert config.pipeline.components.lora_path == "/models/ltx2-refine-lora"
        assert config.pipeline.components.config_root == "/models/ltx2-config"

    def test_torch_compile_kwargs_reach_typed_fields(self):
        config = legacy_from_pretrained_to_config(
            "/models/ltx2", _dreamverse_load_kwargs())
        assert isinstance(config.engine.compile, CompileConfig)
        assert config.engine.compile.enabled is True
        assert config.engine.compile.backend == "inductor"
        assert config.engine.compile.mode == "reduce-overhead"
        assert config.engine.compile.fullgraph is True
        # extras should be empty — all four common kwargs are first class.
        assert config.engine.compile.extras == {}

    def test_uncommon_compile_kwargs_fall_to_extras(self):
        kwargs = _dreamverse_load_kwargs()
        kwargs["torch_compile_kwargs"] = {
            **kwargs["torch_compile_kwargs"],
            "options": {"epilogue_fusion": True},
        }
        config = legacy_from_pretrained_to_config("/models/ltx2", kwargs)
        assert config.engine.compile.extras == {
            "options": {"epilogue_fusion": True},
        }

    def test_vae_tiling_reaches_pipeline_selection(self):
        config = legacy_from_pretrained_to_config(
            "/models/ltx2", _dreamverse_load_kwargs())
        assert config.pipeline.vae_tiling is True

    def test_offload_fields_reach_typed_offload_config(self):
        config = legacy_from_pretrained_to_config(
            "/models/ltx2", _dreamverse_load_kwargs())
        assert config.engine.offload.dit is False
        assert config.engine.offload.vae is False
        assert config.engine.offload.text_encoder is False
        assert config.engine.offload.pin_cpu_memory is True


class TestDreamversePrivateOnlyFields:
    """Dreamverse carries a handful of private-only names (e.g. legacy
    internal aliases). These must NOT silently turn into a public
    compatibility promise — the documented contract is that unknown
    fields land on ``pipeline.experimental`` so integrators see them
    but FastVideo does not promise to preserve them."""

    def test_unknown_kwarg_routes_to_experimental(self):
        kwargs = _dreamverse_load_kwargs()
        kwargs["dreamverse_internal_only_flag"] = "private"
        config = legacy_from_pretrained_to_config("/models/ltx2", kwargs)
        assert config.pipeline.experimental == {
            "dreamverse_internal_only_flag": "private",
        }


class TestDreamverseRequestShape:
    """The per-segment Dreamverse request path mirrors OpenAI's shape
    plus a few LTX-2 knobs. All of them must have a typed home."""

    def test_basic_request_fields_round_trip(self):
        # The request path calls legacy_generate_call_to_request with a
        # prompt + legacy kwargs; verify the typed shape carries them.
        legacy_kwargs = {
            "num_frames": 121,
            "height": 1024,
            "width": 1536,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": 42,
            "fps": 24,
            "negative_prompt": "blurry",
        }
        request = legacy_generate_call_to_request(
            prompt="a fox running",
            sampling_param=None,
            legacy_kwargs=legacy_kwargs,
        )
        request = normalize_generation_request(request)
        assert request.prompt == "a fox running"
        assert request.negative_prompt == "blurry"
        assert request.sampling.num_frames == 121
        assert request.sampling.height == 1024
        assert request.sampling.width == 1536
        assert request.sampling.num_inference_steps == 8
        assert request.sampling.guidance_scale == 1.0
        assert request.sampling.seed == 42
        assert request.sampling.fps == 24

    def test_return_state_reaches_output_config(self):
        """PR 7 added ``output.return_state`` — must survive the legacy
        translation path so Dreamverse callers can opt in."""
        request = GenerationRequest(
            prompt="x",
            output=__import__(
                "fastvideo.api", fromlist=["OutputConfig"]).OutputConfig(
                    return_state=True),
        )
        normalized = normalize_generation_request(request)
        assert normalized.output.return_state is True


class TestDreamverseNoPrivateImports:
    """The public entry points must not force a Dreamverse integrator
    to import from ``fastvideo.pipelines.*`` or other internal paths."""

    @pytest.mark.parametrize(
        "import_path",
        [
            "fastvideo",
            "fastvideo.api",
            "fastvideo.api.compat",  # public in that it's re-exported
        ],
    )
    def test_public_imports_resolve(self, import_path):
        import importlib

        importlib.import_module(import_path)
