# SPDX-License-Identifier: Apache-2.0
"""gpu_pool-style flat-kwarg integration tests.

Mirrors the ``load_kwargs`` dict that the FastVideo-internal
``ui/ltx2-streaming/server/gpu_pool.py`` passes to
``VideoGenerator.from_pretrained(**load_kwargs)`` and asserts that the
public typed ``GeneratorConfig`` surface (introduced across PRs 0-6)
can represent it end-to-end, with no fields silently falling through
to ``pipeline.experimental``.

This is the parity guard PR 7.6 depends on: the public gpu_pool
upstream must be able to construct a typed ``GeneratorConfig`` without
knowing any legacy LTX-2 kwarg name, and downstream Dynamo
(``FastVideoArgGroup``) must be able to do the same.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from fastvideo.api.compat import (
    generator_config_to_fastvideo_args,
    legacy_from_pretrained_to_config,
)


# Mirrors FastVideo-internal/ui/ltx2-streaming/server/gpu_pool.py
# :lines 233-260 (load_kwargs constructed for VideoGenerator.from_pretrained).
# Skipped here because they are opaque Python objects that legitimately
# belong in experimental:
#   - pipeline_config=<PipelineConfig instance>
#   - enable_torch_compile_text_encoder (not in public FastVideoArgs)
GPU_POOL_LOAD_KWARGS = {
    "config_model_path": "/models/ltx2-distilled/config",
    "num_gpus": 1,
    "dit_layerwise_offload": False,
    "use_fsdp_inference": False,
    "dit_cpu_offload": False,
    "vae_cpu_offload": False,
    "text_encoder_cpu_offload": False,
    "pin_cpu_memory": True,
    "ltx2_vae_tiling": False,
    "ltx2_refine_enabled": True,
    "ltx2_refine_upsampler_path": "/models/ltx2-distilled/spatial_upsampler",
    "ltx2_refine_lora_path": "",
    "ltx2_refine_num_inference_steps": 2,
    "ltx2_refine_guidance_scale": 1.0,
    "ltx2_refine_add_noise": True,
    "enable_torch_compile": True,
    "torch_compile_kwargs": {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": False,
    },
}


class TestGpuPoolForwardTranslation:
    """gpu_pool flat kwargs -> typed GeneratorConfig."""

    @pytest.fixture(scope="class")
    def config(self):
        return legacy_from_pretrained_to_config(
            "FastVideo/LTX2-Distilled-Diffusers",
            GPU_POOL_LOAD_KWARGS,
        )

    def test_model_path_set(self, config) -> None:
        assert config.model_path == "FastVideo/LTX2-Distilled-Diffusers"

    def test_engine_basics(self, config) -> None:
        assert config.engine.num_gpus == 1
        assert config.engine.use_fsdp_inference is False

    def test_offload_config(self, config) -> None:
        assert config.engine.offload.dit is False
        assert config.engine.offload.dit_layerwise is False
        assert config.engine.offload.vae is False
        assert config.engine.offload.text_encoder is False
        assert config.engine.offload.pin_cpu_memory is True

    def test_compile_config_typed_fields_extracted(self, config) -> None:
        compile_config = config.engine.compile
        assert compile_config.enabled is True
        assert compile_config.backend == "inductor"
        assert compile_config.fullgraph is True
        assert compile_config.mode == "max-autotune-no-cudagraphs"
        assert compile_config.dynamic is False
        assert compile_config.extras == {}

    def test_vae_tiling_routed_to_pipeline(self, config) -> None:
        assert config.pipeline.vae_tiling is False

    def test_config_model_path_routed_to_components(self, config) -> None:
        assert config.pipeline.components.config_root == "/models/ltx2-distilled/config"

    def test_refine_upsampler_routed_to_components(self, config) -> None:
        assert config.pipeline.components.upsampler_weights == (
            "/models/ltx2-distilled/spatial_upsampler")

    def test_empty_refine_lora_becomes_none(self, config) -> None:
        # gpu_pool passes "" to keep refine LoRA disabled; typed schema
        # treats that as "no LoRA" rather than an empty-string path.
        assert config.pipeline.components.lora_path is None

    def test_refine_preset_overrides(self, config) -> None:
        refine = config.pipeline.preset_overrides.get("refine", {})
        assert refine == {
            "enabled": True,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "add_noise": True,
        }

    def test_no_experimental_leakage(self, config) -> None:
        """Every gpu_pool kwarg should have a typed home — nothing should
        silently fall through to ``pipeline.experimental``."""
        assert config.pipeline.experimental == {}


class TestGpuPoolReverseTranslation:
    """typed GeneratorConfig -> FastVideoArgs kwargs reproduces the
    original gpu_pool flat-kwarg shape.

    This is what lets PR 7.6 wire the public ``gpu_pool`` through
    ``generator_config_to_fastvideo_args`` without the runtime noticing.
    """

    @pytest.fixture
    def args_kwargs(self, monkeypatch):
        from fastvideo import fastvideo_args as fva

        captured: dict[str, object] = {}

        def _capture(**kw):
            captured.update(kw)
            return _Captured(**kw)

        class _Captured:

            def __init__(self, **kw):
                self.kwargs = kw

        monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _capture)

        config = legacy_from_pretrained_to_config(
            "FastVideo/LTX2-Distilled-Diffusers",
            GPU_POOL_LOAD_KWARGS,
        )
        generator_config_to_fastvideo_args(config)
        return captured

    def test_ltx2_refine_flags_reemitted(self, args_kwargs) -> None:
        assert args_kwargs["ltx2_refine_enabled"] is True
        assert args_kwargs["ltx2_refine_add_noise"] is True
        assert args_kwargs["ltx2_refine_num_inference_steps"] == 2
        assert args_kwargs["ltx2_refine_guidance_scale"] == 1.0

    def test_refine_upsampler_path_reemitted(self, args_kwargs) -> None:
        assert args_kwargs["ltx2_refine_upsampler_path"] == (
            "/models/ltx2-distilled/spatial_upsampler")

    def test_config_model_path_reemitted(self, args_kwargs) -> None:
        assert args_kwargs["config_model_path"] == "/models/ltx2-distilled/config"

    def test_torch_compile_kwargs_reassembled(self, args_kwargs) -> None:
        assert args_kwargs["torch_compile_kwargs"] == {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": False,
        }

    def test_vae_tiling_reemitted_with_legacy_name(self, args_kwargs) -> None:
        assert args_kwargs["ltx2_vae_tiling"] is False

    def test_no_stray_refine_dict(self, args_kwargs) -> None:
        """preset_overrides.refine must flatten to ltx2_refine_* kwargs
        rather than landing as a nested ``refine`` kwarg that
        FastVideoArgs doesn't understand."""
        assert "refine" not in args_kwargs


class TestCompileExtrasPreserved:
    """Additional torch.compile kwargs beyond the four typed fields
    round-trip through ``CompileConfig.extras``."""

    def test_extras_preserved(self, monkeypatch) -> None:
        from fastvideo import fastvideo_args as fva

        captured: dict[str, object] = {}

        def _capture(**kw):
            captured.update(kw)

            class _Captured:

                def __init__(self, **kw):
                    self.kwargs = kw

            return _Captured(**kw)

        monkeypatch.setattr(fva.FastVideoArgs, "from_kwargs", _capture)

        kwargs = deepcopy(GPU_POOL_LOAD_KWARGS)
        kwargs["torch_compile_kwargs"] = {
            "backend": "inductor",
            "options": {"triton.cudagraphs": False},
            "disable": False,
        }
        config = legacy_from_pretrained_to_config(
            "FastVideo/LTX2-Distilled-Diffusers", kwargs)
        assert config.engine.compile.backend == "inductor"
        assert config.engine.compile.extras == {
            "options": {"triton.cudagraphs": False},
            "disable": False,
        }

        generator_config_to_fastvideo_args(config)
        assert captured["torch_compile_kwargs"] == {
            "backend": "inductor",
            "options": {"triton.cudagraphs": False},
            "disable": False,
        }
